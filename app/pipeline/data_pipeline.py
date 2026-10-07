# code : pipeline/data_pipeline.py

from __future__ import annotations

import json
import re
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.database.connection import get_engine
from app.llm.fake_client import FakeLLMClient
from app.ml.ml_router import AnalysisMode, MLRouter, MLTask
from app.presentation.result_formatter import ResultFormatter
from app.presentation.result_visualizer import ResultVisualizer
from app.sql.executor import SQLExecutor
from app.sql.generator import SQLGenerator
from app.sql.permissions import (
    build_allowed_columns,
    build_allowed_relationships,
    build_allowed_tables,
)
from app.sql.validator import SQLValidator

# =====================================================================
# CHEMINS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = PROJECT_ROOT / "semantic_catalog.json"
CONTEXT_PATH = PROJECT_ROOT / "semantic_context.txt"

# =====================================================================
# PRÉVISION : SQL FIXE (utilisé par le pipeline classique de repli)
# =====================================================================

DEFAULT_HORIZON_MONTHS = 6

FORECAST_KPIS = {
    "chiffre_affaires": "Chiffre_Affaires",
    "marge": "Marge",
    "quantite": "Quantite",
}

FORECAST_LABELS = {
    "Chiffre_Affaires": "chiffre d'affaires",
    "Marge": "marge",
    "Quantite": "quantité",
}

BREAKDOWN_PATTERN = re.compile(r"\b(?:regions?|produits?|clients?|categories?)\b")


def build_forecast_sql(metric: str | None) -> str:
    """Série mensuelle de l'indicateur (colonne tirée d'une liste fermée)."""
    column = FORECAST_KPIS.get(metric or "", FORECAST_KPIS["chiffre_affaires"])
    return f"""
SELECT
    dd.Annee AS Annee,
    dd.Mois AS Mois,
    SUM(fv.{column}) AS {column}
FROM dbo.Fact_Ventes AS fv
INNER JOIN dbo.Dim_Date AS dd
    ON fv.Date_ID = dd.Date_ID
GROUP BY dd.Annee, dd.Mois
ORDER BY dd.Annee, dd.Mois
""".strip()


def _unaccent(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


class SQLRejectedError(ValueError):
    """Requête refusée par le validateur (avec erreurs et avertissements)."""

    def __init__(self, message, errors=None, warnings=None):
        super().__init__(message)
        self.errors = list(errors or [])
        self.warnings = list(warnings or [])


@contextmanager
def _timed(label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"[PERF] {label}: {time.perf_counter() - start:.3f}s")


# =====================================================================
# PIPELINE RESULT
# =====================================================================


@dataclass
class PipelineResult:
    """Résultat global produit par le DataPipeline ou l'agent."""

    status: str = "ok"
    question: str = ""

    # SQL
    sql: str | None = None
    normalized_sql: str | None = None

    # Résultats
    dataframe: pd.DataFrame | None = None
    formatted_result: Any | None = None
    chart_path: Path | str | None = None

    # ML
    ml_result: Any | None = None
    forecast_result: Any | None = None
    forecast_chart_path: Path | str | None = None
    ml_model_name: str | None = None
    ml_mae: float | None = None
    ml_rmse: float | None = None
    ml_mape: float | None = None
    ml_dataframe: pd.DataFrame | None = None

    # Routage
    analysis_mode: str | None = None
    ml_task: str | None = None
    horizon_months: int | None = None

    # Réponse métier
    explanation: str | None = None
    tables_used: list[str] = field(default_factory=list)
    columns_used: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    # Réponse rédigée par l'assistant (mode conversationnel)
    analysis_text: str | None = None

    # Validation
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Clarification
    clarification_question: str | None = None

    # Couverture temporelle
    data_min_year: int | None = None
    data_max_year: int | None = None

    # Divers
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_forecast(self) -> bool:
        return self.forecast_result is not None

    @property
    def forecast_metrics(self) -> dict[str, Any]:
        metrics = getattr(self.forecast_result, "metrics", {})
        return metrics if isinstance(metrics, dict) else {}

    @property
    def forecast_dataframe(self) -> pd.DataFrame | None:
        forecast = getattr(self.forecast_result, "forecast", None)
        return forecast if isinstance(forecast, pd.DataFrame) else None

    @property
    def historical_dataframe(self) -> pd.DataFrame | None:
        historical = getattr(self.forecast_result, "historical", None)
        return historical if isinstance(historical, pd.DataFrame) else None


# =====================================================================
# DATA PIPELINE
# =====================================================================


class DataPipeline:
    """
    Deux modes :

    - run_chat() : mode conversationnel (agent avec outils, voir data_agent.py).
    - run()      : pipeline classique question -> SQL -> résultat. Sert de
                   repli si l'agent échoue.
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        max_rows: int = 1000,
        max_display_rows: int = 20,
        output_dir: str | Path = "outputs",
    ) -> None:
        self.llm_client = llm_client if llm_client is not None else FakeLLMClient()
        self.max_rows = max_rows
        self.max_display_rows = max_display_rows
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.catalog = self._load_catalog()
        self.semantic_context = self._load_semantic_context()

        self.ml_router = MLRouter()
        self.model_registry = None   # chargé à la demande
        self._agent = None

        self.sql_generator = SQLGenerator(llm_client=self.llm_client)

        self.sql_validator = SQLValidator(
            allowed_tables=build_allowed_tables(self.catalog),
            allowed_columns=build_allowed_columns(self.catalog),
            allowed_relationships=build_allowed_relationships(self.catalog),
        )

        self.sql_executor = SQLExecutor(
            engine=get_engine(),
            max_rows=self.max_rows,
            dry_run=False,
        )

        self.result_formatter = ResultFormatter(max_display_rows=self.max_display_rows)
        self.visualizer = ResultVisualizer(output_dir=self.output_dir)

    # =================================================================
    # MODE CONVERSATIONNEL
    # =================================================================

    def get_agent(self):
        from app.pipeline.data_agent import DataAgent

        if self._agent is None:
            self._agent = DataAgent(self)
        return self._agent

    def run_chat(
        self,
        question: str,
        history: list[dict] | None = None,
        memory: list[dict] | None = None,
        filters: dict[str, str] | None = None,
        focus_region: str | None = None,
    ) -> PipelineResult:
        try:
            return self.get_agent().run(question, history, memory, filters, focus_region)
        except Exception as exc:
            print(f"[WARN] Agent indisponible, repli sur le pipeline classique : {exc}")
            return self.run(question)

    # =================================================================
    # PIPELINE CLASSIQUE
    # =================================================================

    def run(self, question: str) -> PipelineResult:
        total_start = time.perf_counter()

        if not question or not question.strip():
            return PipelineResult(
                status="error", question=question or "",
                errors=["La question est vide."],
            )

        question = question.strip()

        # 1. Routage
        with _timed("Router"):
            routing = self.ml_router.route(question)

        analysis_mode = routing.mode
        ml_task = routing.ml_task
        horizon_months = routing.horizon_months
        mode_value = getattr(analysis_mode, "value", analysis_mode)
        task_value = getattr(ml_task, "value", ml_task)

        print(f"[ROUTING] mode={mode_value}, ml_task={task_value}, horizon={horizon_months}")

        # 1bis. Prévision : chemin dédié (SQL fixe)
        if analysis_mode in (AnalysisMode.ML, AnalysisMode.SQL_ML) and ml_task == MLTask.FORECASTING:
            result = self._run_forecast_flow(
                question=question,
                mode_value=mode_value,
                task_value=task_value,
                horizon_months=horizon_months or DEFAULT_HORIZON_MONTHS,
                metric=getattr(routing, "metric", None),
            )
            print(f"[PERF] TOTAL: {time.perf_counter() - total_start:.3f}s")
            return result

        # 2. Génération SQL
        with _timed("SQL generation"):
            generated = self._generate_sql(question=question)

        if generated.get("status", "ok") != "ok":
            return PipelineResult(
                status="clarification_needed",
                question=question,
                clarification_question=generated.get("clarification_question"),
                analysis_mode=mode_value,
                ml_task=task_value,
            )

        sql = generated.get("sql")

        # 3. Validation
        with _timed("SQL validation"):
            try:
                normalized_sql, sql_warnings = self._validate_sql(sql)
            except SQLRejectedError as exc:
                return PipelineResult(
                    status="validation_error",
                    question=question,
                    sql=sql,
                    errors=exc.errors or [str(exc)],
                    warnings=exc.warnings,
                    analysis_mode=mode_value,
                    ml_task=task_value,
                )

        # 4. Exécution
        with _timed("SQL execution"):
            dataframe = self._execute_sql(sql=normalized_sql)

        if dataframe is None:
            dataframe = pd.DataFrame()

        # 5. Formatage
        with _timed("Formatting"):
            formatted_result = self._format_result(dataframe=dataframe)

        # 6. Graphique
        chart_path = None
        if not dataframe.empty:
            try:
                chart_path = self.visualizer.create_chart(dataframe, filename="result.png")
            except Exception as exc:
                print(f"[WARN] Graphique SQL non créé : {exc}")

        # 7. Couverture temporelle (si aucun résultat)
        data_min_year = data_max_year = None
        if dataframe.empty:
            data_min_year, data_max_year = self._get_data_year_range()

        # 8. Résultat
        result = PipelineResult(
            status="ok",
            question=question,
            sql=sql,
            normalized_sql=normalized_sql,
            dataframe=dataframe,
            formatted_result=formatted_result,
            chart_path=chart_path,
            warnings=sql_warnings,
            explanation=generated.get("explanation"),
            tables_used=generated.get("tables_used", []),
            columns_used=generated.get("columns_used", []),
            assumptions=generated.get("assumptions", []),
            data_min_year=data_min_year,
            data_max_year=data_max_year,
            analysis_mode=mode_value,
            ml_task=task_value,
            horizon_months=horizon_months,
        )

        # 9. Autres tâches ML
        if analysis_mode in (AnalysisMode.ML, AnalysisMode.SQL_ML) and ml_task != MLTask.NONE:
            try:
                with _timed("ML analysis"):
                    ml_result = self._run_ml_analysis(
                        dataframe=dataframe, ml_task=ml_task, horizon_months=horizon_months
                    )
                result = self._attach_ml_result(result=result, ml_result=ml_result, ml_task=ml_task)
            except NotImplementedError as exc:
                result.warnings.append(str(exc))

        print(f"[PERF] TOTAL: {time.perf_counter() - total_start:.3f}s")
        return result

    # =================================================================
    # FLUX PRÉVISION (SQL fixe, sans LLM)
    # =================================================================

    def _run_forecast_flow(
        self,
        question: str,
        mode_value: str,
        task_value: str,
        horizon_months: int,
        metric: str | None = None,
    ) -> PipelineResult:
        column = FORECAST_KPIS.get(metric or "", FORECAST_KPIS["chiffre_affaires"])
        forecast_sql = build_forecast_sql(metric)

        with _timed("Forecast SQL"):
            try:
                normalized_sql, sql_warnings = self._validate_sql(forecast_sql)
            except SQLRejectedError as exc:
                return PipelineResult(
                    status="validation_error",
                    question=question,
                    sql=forecast_sql,
                    errors=exc.errors or [str(exc)],
                    warnings=exc.warnings,
                    analysis_mode=mode_value,
                    ml_task=task_value,
                )
            raw = self._execute_sql(sql=normalized_sql)

        if raw is None or raw.empty:
            return PipelineResult(
                status="error",
                question=question,
                sql=forecast_sql,
                errors=["Aucune donnée historique pour produire une prévision."],
                analysis_mode=mode_value,
                ml_task=task_value,
            )

        history = pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    pd.DataFrame(
                        {
                            "year": raw["Annee"].astype(int),
                            "month": raw["Mois"].astype(int),
                            "day": 1,
                        }
                    )
                ),
                column: pd.to_numeric(raw[column], errors="coerce"),
            }
        )

        warnings = list(sql_warnings)
        if BREAKDOWN_PATTERN.search(_unaccent(question)):
            warnings.append(
                "La prévision porte sur l'ensemble de l'activité : le détail par "
                "région, produit ou client n'est pas disponible dans ce mode."
            )

        result = PipelineResult(
            status="ok",
            question=question,
            sql=forecast_sql,
            normalized_sql=normalized_sql,
            dataframe=history,
            warnings=warnings,
            explanation=f"Série mensuelle ({column}), prévision sur {horizon_months} mois.",
            tables_used=["Fact_Ventes", "Dim_Date"],
            columns_used=[column, "Annee", "Mois"],
            assumptions=[
                f"Indicateur prévu : {FORECAST_LABELS.get(column, column)}.",
                "Périmètre : ensemble de l'activité.",
                "Historique : tous les mois disponibles.",
                f"Horizon : {horizon_months} mois après le dernier mois connu.",
            ],
            analysis_mode=mode_value,
            ml_task=task_value,
            horizon_months=horizon_months,
        )

        try:
            with _timed("ML analysis"):
                ml_result = self._run_ml_analysis(
                    dataframe=history,
                    ml_task=MLTask.FORECASTING,
                    horizon_months=horizon_months,
                )
        except ValueError as exc:
            result.status = "error"
            result.errors = [str(exc)]
            return result

        return self._attach_ml_result(
            result=result, ml_result=ml_result, ml_task=MLTask.FORECASTING
        )

    # =================================================================
    # MACHINE LEARNING
    # =================================================================

    def _run_ml_analysis(
        self,
        dataframe: pd.DataFrame,
        ml_task: MLTask,
        horizon_months: int | None = None,
    ) -> Any:
        if ml_task == MLTask.FORECASTING:
            return self._run_forecasting(dataframe=dataframe, horizon_months=horizon_months)
        if ml_task == MLTask.ANOMALY_DETECTION:
            raise NotImplementedError("La détection d'anomalies n'est pas encore implémentée.")
        if ml_task == MLTask.CLUSTERING:
            raise NotImplementedError("Le clustering n'est pas encore implémenté.")
        if ml_task == MLTask.CLASSIFICATION:
            raise NotImplementedError("La classification n'est pas encore implémentée.")
        return None

    def _run_forecasting(
        self,
        dataframe: pd.DataFrame,
        horizon_months: int | None = None,
    ) -> Any:
        """Prépare une série (colonne de date + une mesure) puis lance le modèle."""
        if dataframe is None or dataframe.empty:
            raise ValueError("Impossible de lancer la prévision : le résultat SQL est vide.")

        df = dataframe.copy()

        date_column = self._find_date_column(df)
        if date_column is None:
            raise ValueError("Impossible d'identifier la colonne de date dans le résultat SQL.")

        value_column = self._find_value_column(df)
        if value_column is None:
            raise ValueError("Impossible d'identifier la colonne de valeur dans le résultat SQL.")

        forecast_df = df[[date_column, value_column]].copy()
        forecast_df = forecast_df.rename(columns={date_column: "Date"})
        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"], errors="coerce")
        forecast_df[value_column] = pd.to_numeric(forecast_df[value_column], errors="coerce")
        forecast_df = forecast_df.dropna(subset=["Date", value_column])

        if forecast_df.empty:
            raise ValueError("Aucune donnée exploitable pour la prévision.")

        forecast_df = (
            forecast_df.sort_values("Date")
            .set_index("Date")
            .resample("MS")[value_column]
            .sum()
            .reset_index()
        )

        horizon = int(horizon_months if horizon_months is not None else DEFAULT_HORIZON_MONTHS)
        if horizon <= 0:
            raise ValueError("L'horizon de prévision doit être supérieur à 0.")

        model = self._get_forecasting_model()
        return model.run(forecast_df, horizon=horizon)

    def _get_forecasting_model(self):
        if self.model_registry is None:
            from app.ml.model_registry import ModelRegistry

            self.model_registry = ModelRegistry()
        return self.model_registry.get_model_for_task(MLTask.FORECASTING)

    def _attach_ml_result(
        self,
        result: PipelineResult,
        ml_result: Any,
        ml_task: MLTask,
    ) -> PipelineResult:
        result.ml_result = ml_result
        if ml_result is None:
            return result

        historical = getattr(ml_result, "historical", None)
        forecast = getattr(ml_result, "forecast", None)

        if ml_task == MLTask.FORECASTING:
            result.forecast_result = ml_result
            result.ml_model_name = getattr(ml_result, "model_name", None)

            metrics = getattr(ml_result, "metrics", {})
            if isinstance(metrics, dict):
                result.ml_mae = self._metric_value(metrics, "mae", "MAE")
                result.ml_rmse = self._metric_value(metrics, "rmse", "RMSE")
                result.ml_mape = self._metric_value(metrics, "mape", "MAPE")

            result.ml_dataframe = self._build_ml_dataframe(historical, forecast)

            try:
                result.forecast_chart_path = self.visualizer.create_forecast_chart(
                    forecast_result=ml_result, filename="forecast.png"
                )
            except Exception as exc:
                result.warnings.append(
                    f"La prévision a été calculée, mais le graphique n'a pas pu être créé : {exc}"
                )
            return result

        # ML générique
        result.ml_model_name = getattr(ml_result, "model_name", None)
        result.ml_mae = getattr(ml_result, "mae", None)
        result.ml_rmse = getattr(ml_result, "rmse", None)
        result.ml_mape = getattr(ml_result, "mape", None)

        if isinstance(forecast, pd.DataFrame) or isinstance(historical, pd.DataFrame):
            result.ml_dataframe = self._build_ml_dataframe(historical, forecast)

        return result

    @staticmethod
    def _build_ml_dataframe(historical: Any = None, forecast: Any = None) -> pd.DataFrame | None:
        frames: list[pd.DataFrame] = []

        if isinstance(historical, pd.DataFrame):
            hist = historical.copy()
            hist["_type"] = "historique"
            frames.append(hist)

        if isinstance(forecast, pd.DataFrame):
            fc = forecast.copy()
            fc["_type"] = "prévision"
            frames.append(fc)

        if not frames:
            return None

        try:
            return pd.concat(frames, ignore_index=True)
        except Exception:
            return frames[0]

    @staticmethod
    def _metric_value(metrics: dict, *keys: str) -> float | None:
        for key in keys:
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    return None
        return None

    # =================================================================
    # DÉTECTION DE COLONNES
    # =================================================================

    @staticmethod
    def _find_date_column(dataframe: pd.DataFrame) -> str | None:
        for column in ("Date", "date", "DATE", "Date_Mois", "date_mois"):
            if column in dataframe.columns:
                return column

        for column in dataframe.columns:
            if pd.api.types.is_datetime64_any_dtype(dataframe[column]):
                return column

        # Colonnes numériques ignorées : 1..12 ou 2024 deviendraient des dates de 1970
        for column in dataframe.columns:
            if pd.api.types.is_numeric_dtype(dataframe[column]):
                continue
            converted = pd.to_datetime(dataframe[column], errors="coerce")
            if converted.notna().mean() >= 0.8:
                return column

        return None

    @staticmethod
    def _find_value_column(dataframe: pd.DataFrame) -> str | None:
        preferred = [
            "Chiffre_Affaires", "chiffre_affaires", "Chiffre d'affaires",
            "chiffre d'affaires", "CA", "ca", "Revenue", "revenue",
            "Montant", "montant", "Total", "total",
        ]
        for column in preferred:
            if column in dataframe.columns:
                return column

        numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
        if len(numeric_columns) == 1:
            return numeric_columns[0]

        keywords = ["ca", "chiffre", "affaire", "revenue", "sales", "montant", "total", "amount"]
        for column in dataframe.columns:
            name = str(column).lower()
            if any(k in name for k in keywords):
                return column

        return numeric_columns[0] if numeric_columns else None

    # =================================================================
    # COUVERTURE TEMPORELLE
    # =================================================================

    def _get_data_year_range(self) -> tuple[int | None, int | None]:
        try:
            query = """
            SELECT MIN(dd.Annee) AS Min_Year, MAX(dd.Annee) AS Max_Year
            FROM dbo.Fact_Ventes AS fv
            INNER JOIN dbo.Dim_Date AS dd ON fv.Date_ID = dd.Date_ID
            """
            dataframe = pd.read_sql(query, get_engine())
            if dataframe.empty:
                return None, None

            low = dataframe.iloc[0]["Min_Year"]
            high = dataframe.iloc[0]["Max_Year"]
            if pd.isna(low) or pd.isna(high):
                return None, None

            return int(low), int(high)
        except Exception:
            return None, None

    # =================================================================
    # SQL : génération, validation, exécution (utilisés aussi par l'agent)
    # =================================================================

    def _generate_sql(self, question: str) -> dict[str, Any]:
        generated = self.sql_generator.generate(
            question=question,
            semantic_context=self.semantic_context,
        )

        if isinstance(generated, dict):
            return generated

        return {
            "sql": getattr(generated, "sql", None),
            "normalized_sql": getattr(generated, "normalized_sql", None),
            "tables_used": getattr(generated, "tables_used", []),
            "columns_used": getattr(generated, "columns_used", []),
            "assumptions": getattr(generated, "assumptions", []),
        }

    def _validate_sql(self, sql: str | None) -> tuple[str, list[str]]:
        """Valide le SQL. Retourne (sql_normalisé, avertissements)."""
        if not sql:
            raise SQLRejectedError("Aucune requête SQL n'a été générée.")

        validation = self.sql_validator.validate(sql)
        is_valid = getattr(validation, "is_valid", validation)

        if is_valid is False:
            raise SQLRejectedError(
                "La requête SQL a été rejetée par le validateur.",
                errors=getattr(validation, "errors", None),
                warnings=getattr(validation, "warnings", None),
            )

        normalized = getattr(validation, "normalized_sql", None) or sql
        warnings = list(getattr(validation, "warnings", None) or [])
        return normalized, warnings

    def _execute_sql(self, sql: str) -> pd.DataFrame:
        if self.sql_executor is None:
            raise RuntimeError("sql_executor n'est pas configuré dans DataPipeline.")

        result = self.sql_executor.execute(sql)

        if isinstance(result, pd.DataFrame):
            return result

        if isinstance(result, tuple):
            for item in result:
                if isinstance(item, pd.DataFrame):
                    return item

        raise TypeError("Le SQL executor doit retourner un pandas.DataFrame.")

    def _format_result(self, dataframe: pd.DataFrame) -> Any:
        if self.result_formatter is None:
            return dataframe
        if hasattr(self.result_formatter, "format"):
            return self.result_formatter.format(dataframe)
        if hasattr(self.result_formatter, "format_result"):
            return self.result_formatter.format_result(dataframe)
        return dataframe

    # =================================================================
    # CATALOGUE ET CONTEXTE
    # =================================================================

    @staticmethod
    def _load_catalog() -> dict[str, Any]:
        if not CATALOG_PATH.exists():
            return {}
        try:
            with CATALOG_PATH.open("r", encoding="utf-8") as file:
                data = json.load(file)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _load_semantic_context() -> str:
        if not CONTEXT_PATH.exists():
            return ""
        try:
            return CONTEXT_PATH.read_text(encoding="utf-8")
        except OSError:
            return ""
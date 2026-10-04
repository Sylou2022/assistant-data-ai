# code : pipeline/data_pipeline.py

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json
import re
import time
import unicodedata
import pandas as pd

from app.database.connection import get_engine

from app.llm.fake_client import FakeLLMClient

from app.ml.ml_router import (
    AnalysisMode,
    MLTask,
    MLRouter,
)

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
# PRÉVISION : SQL FIXE (sans LLM)
# =====================================================================

DEFAULT_HORIZON_MONTHS = 6

# Indicateurs que l'on sait prévoir : clé du routeur -> colonne de Fact_Ventes.
FORECAST_KPIS = {
    "chiffre_affaires": "Chiffre_Affaires",
    "marge": "Marge",
    "quantite": "Quantite",
}


def build_forecast_sql(metric: str | None) -> str:
    """
    Série mensuelle de l'indicateur demandé (SQL fixe, sans LLM).

    Seule la colonne vient d'une liste fermée (FORECAST_KPIS) : aucun texte
    de l'utilisateur n'entre dans la requête. Elle passe quand même par le
    validateur, comme toute autre requête.
    """
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


FORECAST_SQL = build_forecast_sql("chiffre_affaires")

# Libellés affichés à l'utilisateur
FORECAST_LABELS = {
    "Chiffre_Affaires": "chiffre d'affaires",
    "Marge": "marge",
    "Quantite": "quantité",
}

# Mots indiquant un détail (région, produit...) que la prévision ne gère pas encore
BREAKDOWN_PATTERN = re.compile(
    r"\b(?:regions?|produits?|clients?|categories?)\b"
)


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
    """Affiche le temps passé dans une étape : [PERF] étape: 0.123s"""
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
    """
    Résultat global produit par le DataPipeline.
    """

    # État général
    status: str = "ok"

    # Question
    question: str = ""

    # SQL
    sql: str | None = None
    normalized_sql: str | None = None

    # Résultats SQL
    dataframe: pd.DataFrame | None = None
    formatted_result: Any | None = None
    chart_path: Path | str | None = None

    # Résultat ML générique
    ml_result: Any | None = None

    # Forecasting
    forecast_result: Any | None = None
    forecast_chart_path: Path | str | None = None

    # Métadonnées ML
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

    tables_used: list[str] = field(
        default_factory=list
    )

    columns_used: list[str] = field(
        default_factory=list
    )

    assumptions: list[str] = field(
        default_factory=list
    )

    # Validation
    warnings: list[str] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )

    # Clarification
    clarification_question: str | None = None

    # Couverture temporelle
    data_min_year: int | None = None
    data_max_year: int | None = None

    # Métadonnées complémentaires
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # =========================================================
    # PROPRIÉTÉS FORECAST
    # =========================================================

    @property
    def has_forecast(self) -> bool:
        return self.forecast_result is not None

    @property
    def forecast_metrics(self) -> dict[str, Any]:
        if self.forecast_result is None:
            return {}

        metrics = getattr(
            self.forecast_result,
            "metrics",
            {},
        )

        if isinstance(metrics, dict):
            return metrics

        return {}

    @property
    def forecast_dataframe(
        self,
    ) -> pd.DataFrame | None:
        if self.forecast_result is None:
            return None

        forecast = getattr(
            self.forecast_result,
            "forecast",
            None,
        )

        if isinstance(forecast, pd.DataFrame):
            return forecast

        return None

    @property
    def historical_dataframe(
        self,
    ) -> pd.DataFrame | None:
        if self.forecast_result is None:
            return None

        historical = getattr(
            self.forecast_result,
            "historical",
            None,
        )

        if isinstance(historical, pd.DataFrame):
            return historical

        return None


# =====================================================================
# DATA PIPELINE
# =====================================================================


class DataPipeline:
    """
    Pipeline principal :

        Question
            ↓
        ML Router
            ↓
        ┌───────────────┬──────────────────────────────┐
        │ Prévision     │ Autres questions             │
        │ SQL fixe      │ LLM → SQL → validation       │
        │ + validation  │                              │
        └───────┬───────┴───────────────┬──────────────┘
                ↓                       ↓
            Exécution SQL (lecture seule)
                ↓
        éventuellement ML
                ↓
        PipelineResult

    Le ML est chargé uniquement lorsqu'il est réellement nécessaire.
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        max_rows: int = 1000,
        max_display_rows: int = 20,
        output_dir: str | Path = "outputs",
    ) -> None:

        # LLM
        self.llm_client = (
            llm_client
            if llm_client is not None
            else FakeLLMClient()
        )

        # Configuration
        self.max_rows = max_rows
        self.max_display_rows = max_display_rows
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Catalogue sémantique
        self.catalog = self._load_catalog()

        self.semantic_context = (
            self._load_semantic_context()
        )

        # ML Router
        self.ml_router = MLRouter()

        # Model Registry (chargé plus tard)
        self.model_registry = None

        # SQL Generator
        self.sql_generator = SQLGenerator(
            llm_client=self.llm_client
        )

        # Permissions SQL
        allowed_tables = build_allowed_tables(
            self.catalog
        )

        allowed_columns = build_allowed_columns(
            self.catalog
        )

        allowed_relationships = (
            build_allowed_relationships(
                self.catalog
            )
        )

        # SQL Validator
        self.sql_validator = SQLValidator(
            allowed_tables=allowed_tables,
            allowed_columns=allowed_columns,
            allowed_relationships=allowed_relationships,
        )

        # SQL Executor
        self.sql_executor = SQLExecutor(
            engine=get_engine(),
            max_rows=self.max_rows,
            dry_run=False,
        )

        # Result Formatter
        self.result_formatter = ResultFormatter(
            max_display_rows=self.max_display_rows
        )

        # Visualizer
        self.visualizer = ResultVisualizer(
            output_dir=self.output_dir
        )

    # =================================================================
    # RUN
    # =================================================================

    def run(
        self,
        question: str,
    ) -> PipelineResult:

        total_start = time.perf_counter()

        if not question or not question.strip():
            return PipelineResult(
                status="error",
                question=question or "",
                errors=["La question est vide."],
            )

        question = question.strip()

        # =============================================================
        # 1. ROUTAGE
        # =============================================================

        with _timed("Router"):
            routing_decision = self.ml_router.route(question)

        analysis_mode = routing_decision.mode
        ml_task = routing_decision.ml_task
        horizon_months = routing_decision.horizon_months

        mode_value = getattr(analysis_mode, "value", analysis_mode)
        task_value = getattr(ml_task, "value", ml_task)

        print(
            f"[ROUTING] mode={mode_value}, "
            f"ml_task={task_value}, "
            f"horizon={horizon_months}"
        )

        # =============================================================
        # 1bis. PRÉVISION : chemin dédié (SQL fixe, pas de LLM)
        # =============================================================

        if (
            analysis_mode in (AnalysisMode.ML, AnalysisMode.SQL_ML)
            and ml_task == MLTask.FORECASTING
        ):
            result = self._run_forecast_flow(
                question=question,
                mode_value=mode_value,
                task_value=task_value,
                horizon_months=horizon_months or DEFAULT_HORIZON_MONTHS,
                metric=getattr(routing_decision, "metric", None),
            )

            print(
                f"[PERF] TOTAL: "
                f"{time.perf_counter() - total_start:.3f}s"
            )

            return result

        # =============================================================
        # 2. GÉNÉRATION SQL
        # =============================================================

        with _timed("SQL generation"):
            generated = self._generate_sql(
                question=question,
            )

        # Le générateur peut demander une précision : on la renvoie
        # telle quelle à l'interface au lieu de lever une erreur.
        if generated.get("status", "ok") != "ok":
            return PipelineResult(
                status="clarification_needed",
                question=question,
                clarification_question=generated.get(
                    "clarification_question"
                ),
                analysis_mode=mode_value,
                ml_task=task_value,
            )

        sql = generated.get("sql")

        # =============================================================
        # 3. VALIDATION SQL
        # =============================================================

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

        # =============================================================
        # 4. EXÉCUTION SQL
        # =============================================================

        with _timed("SQL execution"):
            dataframe = self._execute_sql(
                sql=normalized_sql,
            )

        if dataframe is None:
            dataframe = pd.DataFrame()

        # =============================================================
        # 5. FORMATAGE
        # =============================================================

        with _timed("Formatting"):
            formatted_result = self._format_result(
                dataframe=dataframe,
            )

        # =============================================================
        # 6. GRAPHIQUE SQL
        #
        # Le paramètre de create_chart s'appelle `df` : l'ancien appel
        # avec `dataframe=` levait une TypeError avalée par le except.
        # =============================================================

        chart_path = None

        if not dataframe.empty:
            try:
                chart_path = self.visualizer.create_chart(
                    dataframe,
                    filename="result.png",
                )
            except Exception as exc:
                print(f"[WARN] Graphique SQL non créé : {exc}")

        # =============================================================
        # 7. COUVERTURE TEMPORELLE (seulement si aucun résultat)
        # =============================================================

        data_min_year = None
        data_max_year = None

        if dataframe.empty:
            (
                data_min_year,
                data_max_year,
            ) = self._get_data_year_range()

        # =============================================================
        # 8. PIPELINE RESULT
        # =============================================================

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

        # =============================================================
        # 9. AUTRES TÂCHES ML (anomalies, clustering...)
        #
        # Le ML n'est exécuté QUE si le router l'a demandé.
        # =============================================================

        if (
            analysis_mode in (AnalysisMode.ML, AnalysisMode.SQL_ML)
            and ml_task != MLTask.NONE
        ):
            try:
                with _timed("ML analysis"):
                    ml_result = self._run_ml_analysis(
                        dataframe=dataframe,
                        ml_task=ml_task,
                        horizon_months=horizon_months,
                    )

                result = self._attach_ml_result(
                    result=result,
                    ml_result=ml_result,
                    ml_task=ml_task,
                )

            except NotImplementedError as exc:
                result.warnings.append(str(exc))

        print(
            f"[PERF] TOTAL: "
            f"{time.perf_counter() - total_start:.3f}s"
        )

        return result

    # =================================================================
    # FLUX PRÉVISION
    # =================================================================

    def _run_forecast_flow(
        self,
        question: str,
        mode_value: str,
        task_value: str,
        horizon_months: int,
        metric: str | None = None,
    ) -> PipelineResult:
        """
        Série mensuelle fixe -> validateur -> exécution -> modèle.

        Aucun appel LLM : plus de questions de clarification en boucle
        et plus de colonnes de date mal interprétées.
        """

        column = FORECAST_KPIS.get(
            metric or "", FORECAST_KPIS["chiffre_affaires"]
        )
        forecast_sql = build_forecast_sql(metric)

        with _timed("Forecast SQL"):
            try:
                normalized_sql, sql_warnings = self._validate_sql(
                    forecast_sql
                )

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

            raw = self._execute_sql(
                sql=normalized_sql,
            )

        if raw is None or raw.empty:
            return PipelineResult(
                status="error",
                question=question,
                sql=forecast_sql,
                errors=["Aucune donnée historique pour produire une prévision."],
                analysis_mode=mode_value,
                ml_task=task_value,
            )

        # La date est construite côté pandas : le SQL reste simple
        # (pas de fonction) et donc plus sûr pour le validateur.
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
                column: pd.to_numeric(
                    raw[column],
                    errors="coerce",
                ),
            }
        )

        warnings = list(sql_warnings)

        if BREAKDOWN_PATTERN.search(_unaccent(question)):
            warnings.append(
                "La prévision porte sur l'ensemble de l'activité : "
                "le détail par région, produit ou client n'est pas "
                "encore disponible."
            )

        result = PipelineResult(
            status="ok",
            question=question,
            sql=forecast_sql,
            normalized_sql=normalized_sql,
            dataframe=history,
            warnings=warnings,
            explanation=(
                f"Série mensuelle ({column}), "
                f"prévision sur {horizon_months} mois."
            ),
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
            # Message métier déjà en français (historique trop court,
            # horizon trop grand...) : on le transmet à l'interface.
            result.status = "error"
            result.errors = [str(exc)]
            return result

        return self._attach_ml_result(
            result=result,
            ml_result=ml_result,
            ml_task=MLTask.FORECASTING,
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
        """
        Exécute l'analyse ML demandée.
        """

        if ml_task == MLTask.FORECASTING:

            return self._run_forecasting(
                dataframe=dataframe,
                horizon_months=horizon_months,
            )

        if ml_task == MLTask.ANOMALY_DETECTION:

            raise NotImplementedError(
                "La détection d'anomalies "
                "n'est pas encore implémentée."
            )

        if ml_task == MLTask.CLUSTERING:

            raise NotImplementedError(
                "Le clustering "
                "n'est pas encore implémenté."
            )

        if ml_task == MLTask.CLASSIFICATION:

            raise NotImplementedError(
                "La classification "
                "n'est pas encore implémentée."
            )

        return None

    # =================================================================
    # FORECASTING
    # =================================================================

    def _run_forecasting(
        self,
        dataframe: pd.DataFrame,
        horizon_months: int | None = None,
    ) -> Any:
        """
        Prépare le DataFrame SQL puis exécute
        le modèle de forecasting.

        Le modèle attend une colonne Date et une colonne numérique
        (Chiffre_Affaires, Marge ou Quantite).
        """

        if dataframe is None or dataframe.empty:

            raise ValueError(
                "Impossible de lancer le forecasting : "
                "le résultat SQL est vide."
            )

        df = dataframe.copy()

        # Recherche date
        date_column = self._find_date_column(df)

        if date_column is None:

            raise ValueError(
                "Impossible d'identifier la colonne "
                "de date dans le résultat SQL."
            )

        # Recherche valeur
        value_column = self._find_value_column(df)

        if value_column is None:

            raise ValueError(
                "Impossible d'identifier la colonne "
                "de chiffre d'affaires dans le résultat SQL."
            )

        # Normalisation
        forecast_df = df[
            [
                date_column,
                value_column,
            ]
        ].copy()

        # Le nom de l'indicateur (Chiffre_Affaires, Marge, Quantite)
        # est conservé : il sert aux libellés et aux unités de l'interface.
        forecast_df = forecast_df.rename(
            columns={date_column: "Date"}
        )

        forecast_df["Date"] = pd.to_datetime(
            forecast_df["Date"],
            errors="coerce",
        )

        forecast_df[value_column] = (
            pd.to_numeric(
                forecast_df[value_column],
                errors="coerce",
            )
        )

        # Suppression des lignes invalides
        forecast_df = forecast_df.dropna(
            subset=[
                "Date",
                value_column,
            ]
        )

        if forecast_df.empty:

            raise ValueError(
                "Aucune donnée exploitable "
                "pour le forecasting."
            )

        # Agrégation mensuelle
        forecast_df = (
            forecast_df
            .sort_values("Date")
            .set_index("Date")
            .resample("MS")[
                value_column
            ]
            .sum()
            .reset_index()
        )

        # Horizon
        if horizon_months is None:
            horizon_months = DEFAULT_HORIZON_MONTHS

        horizon_months = int(
            horizon_months
        )

        if horizon_months <= 0:

            raise ValueError(
                "L'horizon de forecasting "
                "doit être supérieur à 0."
            )

        # Modèle
        model = self._get_forecasting_model()

        # Exécution
        forecast_result = model.run(
            forecast_df,
            horizon=horizon_months,
        )

        return forecast_result

    # =================================================================
    # MODEL REGISTRY
    # =================================================================

    def _get_forecasting_model(self):
        """
        Charge le ModelRegistry uniquement lorsqu'un
        forecasting est réellement demandé.
        """

        if self.model_registry is None:

            from app.ml.model_registry import ModelRegistry

            self.model_registry = ModelRegistry()

        return self.model_registry.get_model_for_task(
            MLTask.FORECASTING
        )

    # =================================================================
    # ATTACH ML RESULT
    # =================================================================

    def _attach_ml_result(
        self,
        result: PipelineResult,
        ml_result: Any,
        ml_task: MLTask,
    ) -> PipelineResult:
        """
        Injecte le résultat ML dans PipelineResult.
        """

        result.ml_result = ml_result

        if ml_result is None:
            return result

        # =============================================================
        # FORECASTING
        # =============================================================

        if ml_task == MLTask.FORECASTING:

            result.forecast_result = ml_result

            result.ml_model_name = getattr(
                ml_result,
                "model_name",
                None,
            )

            metrics = getattr(
                ml_result,
                "metrics",
                {},
            )

            if isinstance(metrics, dict):

                result.ml_mae = (
                    self._metric_value(
                        metrics,
                        "mae",
                        "MAE",
                    )
                )

                result.ml_rmse = (
                    self._metric_value(
                        metrics,
                        "rmse",
                        "RMSE",
                    )
                )

                result.ml_mape = (
                    self._metric_value(
                        metrics,
                        "mape",
                        "MAPE",
                    )
                )

            historical = getattr(
                ml_result,
                "historical",
                None,
            )

            forecast = getattr(
                ml_result,
                "forecast",
                None,
            )

            result.ml_dataframe = (
                self._build_ml_dataframe(
                    historical=historical,
                    forecast=forecast,
                )
            )

            try:

                result.forecast_chart_path = (
                    self.visualizer.create_forecast_chart(
                        forecast_result=ml_result,
                        filename="forecast.png",
                    )
                )

            except Exception as exc:

                result.warnings.append(
                    "Le forecasting a été calculé, "
                    "mais le graphique n'a pas pu être créé : "
                    f"{exc}"
                )

            return result

        # =============================================================
        # ML GÉNÉRIQUE
        # =============================================================

        result.ml_model_name = getattr(
            ml_result,
            "model_name",
            None,
        )

        result.ml_mae = getattr(
            ml_result,
            "mae",
            None,
        )

        result.ml_rmse = getattr(
            ml_result,
            "rmse",
            None,
        )

        result.ml_mape = getattr(
            ml_result,
            "mape",
            None,
        )

        forecast = getattr(
            ml_result,
            "forecast",
            None,
        )

        historical = getattr(
            ml_result,
            "historical",
            None,
        )

        if (
            isinstance(
                forecast,
                pd.DataFrame,
            )
            or isinstance(
                historical,
                pd.DataFrame,
            )
        ):

            result.ml_dataframe = (
                self._build_ml_dataframe(
                    historical=historical,
                    forecast=forecast,
                )
            )

        return result

    # =================================================================
    # BUILD ML DATAFRAME
    # =================================================================

    def _build_ml_dataframe(
        self,
        historical: Any = None,
        forecast: Any = None,
    ) -> pd.DataFrame | None:
        """
        Construit un DataFrame combinant historique
        et prévisions.
        """

        frames: list[pd.DataFrame] = []

        if isinstance(
            historical,
            pd.DataFrame,
        ):

            hist = historical.copy()

            hist["_type"] = "historique"

            frames.append(hist)

        if isinstance(
            forecast,
            pd.DataFrame,
        ):

            fc = forecast.copy()

            fc["_type"] = "prévision"

            frames.append(fc)

        if not frames:
            return None

        try:

            return pd.concat(
                frames,
                ignore_index=True,
            )

        except Exception:

            return frames[0]

    # =================================================================
    # METRIC VALUE
    # =================================================================

    @staticmethod
    def _metric_value(
        metrics: dict,
        *keys: str,
    ) -> float | None:
        for key in keys:

            if key in metrics:

                value = metrics[key]

                try:
                    return float(value)

                except (
                    TypeError,
                    ValueError,
                ):
                    return None

        return None

    # =================================================================
    # DATE COLUMN
    # =================================================================

    @staticmethod
    def _find_date_column(
        dataframe: pd.DataFrame,
    ) -> str | None:
        """
        « Mois » et « Annee » ne sont plus des noms prioritaires et les
        colonnes numériques sont ignorées : des entiers (1 à 12, 2024)
        convertis par pandas deviennent des dates de 1970.
        """

        for column in (
            "Date",
            "date",
            "DATE",
            "Date_Mois",
            "date_mois",
        ):

            if column in dataframe.columns:
                return column

        for column in dataframe.columns:

            if pd.api.types.is_datetime64_any_dtype(
                dataframe[column]
            ):
                return column

        for column in dataframe.columns:

            if pd.api.types.is_numeric_dtype(
                dataframe[column]
            ):
                continue

            converted = pd.to_datetime(
                dataframe[column],
                errors="coerce",
            )

            valid_ratio = (
                converted.notna().mean()
            )

            if valid_ratio >= 0.8:
                return column

        return None

    # =================================================================
    # VALUE COLUMN
    # =================================================================

    @staticmethod
    def _find_value_column(
        dataframe: pd.DataFrame,
    ) -> str | None:

        preferred_names = [
            "Chiffre_Affaires",
            "chiffre_affaires",
            "Chiffre d'affaires",
            "chiffre d'affaires",
            "CA",
            "ca",
            "Revenue",
            "revenue",
            "Montant",
            "montant",
            "Total",
            "total",
        ]

        for column in preferred_names:

            if column in dataframe.columns:
                return column

        numeric_columns = (
            dataframe
            .select_dtypes(
                include="number"
            )
            .columns
            .tolist()
        )

        if len(numeric_columns) == 1:
            return numeric_columns[0]

        keywords = [
            "ca",
            "chiffre",
            "affaire",
            "revenue",
            "sales",
            "montant",
            "total",
            "amount",
        ]

        for column in dataframe.columns:

            name = str(column).lower()

            if any(
                keyword in name
                for keyword in keywords
            ):
                return column

        if numeric_columns:
            return numeric_columns[0]

        return None

    # =================================================================
    # COUVERTURE TEMPORELLE
    # =================================================================

    def _get_data_year_range(
        self,
    ) -> tuple[int | None, int | None]:
        """
        Première et dernière année réellement disponibles en base.
        Utilisée uniquement quand une requête ne renvoie aucun résultat.
        """

        try:

            query = """
            SELECT
                MIN(dd.Annee) AS Min_Year,
                MAX(dd.Annee) AS Max_Year
            FROM dbo.Fact_Ventes AS fv
            INNER JOIN dbo.Dim_Date AS dd
                ON fv.Date_ID = dd.Date_ID
            """

            dataframe = pd.read_sql(
                query,
                get_engine(),
            )

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
    # SQL GENERATION
    # =================================================================

    def _generate_sql(
        self,
        question: str,
    ) -> dict[str, Any]:
        """
        Génère le SQL avec le SQLGenerator
        de l'application.
        """

        generated = self.sql_generator.generate(
            question=question,
            semantic_context=self.semantic_context,
        )

        if isinstance(
            generated,
            dict,
        ):
            return generated

        return {
            "sql": getattr(
                generated,
                "sql",
                None,
            ),
            "normalized_sql": getattr(
                generated,
                "normalized_sql",
                None,
            ),
            "tables_used": getattr(
                generated,
                "tables_used",
                [],
            ),
            "columns_used": getattr(
                generated,
                "columns_used",
                [],
            ),
            "assumptions": getattr(
                generated,
                "assumptions",
                [],
            ),
        }

    # =================================================================
    # SQL VALIDATION
    # =================================================================

    def _validate_sql(
        self,
        sql: str | None,
    ) -> tuple[str, list[str]]:
        """
        Valide le SQL. Retourne (sql_normalisé, avertissements).

        Le SQL exécuté est celui normalisé par le validateur, comme dans
        la version précédente du pipeline.
        """

        if not sql:
            raise SQLRejectedError(
                "Aucune requête SQL n'a été générée."
            )

        validation = self.sql_validator.validate(
            sql
        )

        is_valid = getattr(
            validation,
            "is_valid",
            validation,
        )

        if is_valid is False:

            raise SQLRejectedError(
                "La requête SQL a été rejetée "
                "par le validateur.",
                errors=getattr(
                    validation,
                    "errors",
                    None,
                ),
                warnings=getattr(
                    validation,
                    "warnings",
                    None,
                ),
            )

        normalized = (
            getattr(
                validation,
                "normalized_sql",
                None,
            )
            or sql
        )

        warnings = list(
            getattr(
                validation,
                "warnings",
                None,
            )
            or []
        )

        return normalized, warnings

    # =================================================================
    # SQL EXECUTION
    # =================================================================

    def _execute_sql(
        self,
        sql: str,
    ) -> pd.DataFrame:
        """
        Exécute le SQL.
        """

        if self.sql_executor is None:

            raise RuntimeError(
                "sql_executor n'est pas configuré "
                "dans DataPipeline."
            )

        result = self.sql_executor.execute(
            sql
        )

        if isinstance(
            result,
            pd.DataFrame,
        ):
            return result

        if isinstance(
            result,
            tuple,
        ):

            for item in result:

                if isinstance(
                    item,
                    pd.DataFrame,
                ):
                    return item

        raise TypeError(
            "Le SQL executor doit retourner "
            "un pandas.DataFrame."
        )

    # =================================================================
    # RESULT FORMAT
    # =================================================================

    def _format_result(
        self,
        dataframe: pd.DataFrame,
    ) -> Any:
        """
        Formate le résultat SQL.
        """

        if self.result_formatter is None:
            return dataframe

        if hasattr(
            self.result_formatter,
            "format",
        ):

            return self.result_formatter.format(
                dataframe
            )

        if hasattr(
            self.result_formatter,
            "format_result",
        ):

            return (
                self.result_formatter.format_result(
                    dataframe
                )
            )

        return dataframe

    # =================================================================
    # LOAD CATALOG
    # =================================================================

    @staticmethod
    def _load_catalog() -> dict[str, Any]:
        """
        Charge le catalogue sémantique JSON.
        """

        if not CATALOG_PATH.exists():
            return {}

        try:

            with CATALOG_PATH.open(
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            if isinstance(data, dict):
                return data

            return {}

        except (
            OSError,
            json.JSONDecodeError,
        ):

            return {}

    # =================================================================
    # LOAD SEMANTIC CONTEXT
    # =================================================================

    @staticmethod
    def _load_semantic_context() -> str:
        """
        Charge le contexte sémantique texte.
        """

        if not CONTEXT_PATH.exists():
            return ""

        try:

            return CONTEXT_PATH.read_text(
                encoding="utf-8"
            )

        except OSError:

            return ""
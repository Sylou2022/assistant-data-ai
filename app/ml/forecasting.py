# code : ml/forecasting.py

"""Forecasting mensuel de l'Assistant Data IA.

API publique : ForecastingModel().run(dataframe, horizon=6) -> ForecastResult

La sélection compare plusieurs modèles sur un backtest glissant et garde
celui dont la MAE est la plus faible. La validation reste volontairement
bornée afin que l'interface ne soit pas ralentie par un grand nombre de
ré-ajustements statistiques.
"""
from __future__ import annotations

from .forecasting_xgboost import (
    XGBOOST_AVAILABLE,
    forecast_xgboost,
)

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

logger = logging.getLogger(__name__)

SEASON = 12
Z_80 = 1.2816
RELIABILITY_LEVELS = ("faible", "moyenne", "bonne")


@dataclass
class ForecastResult:
    historical: pd.DataFrame
    forecast: pd.DataFrame
    metrics: dict[str, float]
    horizon_months: int
    model_name: str
    date_column: str = "Date"
    value_column: str = "Chiffre_Affaires"
    reliability: str = "faible"
    selection_reason: str = ""
    baseline_name: str = ""
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    candidates: list[dict] = field(default_factory=list)
    quality_warnings: list[str] = field(default_factory=list)
    backtest_folds: int = 0
    backtest_horizon: int = 0
    interval_level: float = 0.8

    @property
    def result(self) -> pd.DataFrame:
        historical = self.historical.copy()
        forecast = self.forecast.copy()
        historical["Type"] = "historique"
        forecast["Type"] = "prévision"
        return pd.concat([historical, forecast], ignore_index=True)


def _future_index(train: pd.Series, h: int) -> pd.DatetimeIndex:
    return pd.date_range(
        start=train.index[-1] + pd.offsets.MonthBegin(1),
        periods=h,
        freq="MS",
    )


def _fc_seasonal_naive(train: pd.Series, h: int) -> np.ndarray:
    values = train.to_numpy(dtype=float)
    if len(values) < SEASON:
        return np.repeat(values[-1], h)
    return np.array(
        [values[len(values) - SEASON + (i % SEASON)] for i in range(h)],
        dtype=float,
    )


def _fc_linear(train: pd.Series, h: int) -> np.ndarray:
    n = len(train)
    slope, intercept = np.polyfit(
        np.arange(n, dtype=float), train.to_numpy(dtype=float), 1
    )
    return intercept + slope * np.arange(n, n + h, dtype=float)


def _fc_linear_seasonal(train: pd.Series, h: int) -> np.ndarray:
    n = len(train)
    values = train.to_numpy(dtype=float)
    slope, intercept = np.polyfit(np.arange(n, dtype=float), values, 1)
    trend = intercept + slope * np.arange(n, dtype=float)
    profile = pd.Series(values - trend).groupby(train.index.month).mean()
    profile = profile - profile.mean()
    future_trend = intercept + slope * np.arange(n, n + h, dtype=float)
    future_months = _future_index(train, h).month
    seasonal = np.array([profile.get(m, 0.0) for m in future_months], dtype=float)
    return future_trend + seasonal


def _fc_holt_damped(train: pd.Series, h: int) -> np.ndarray:
    fit = ExponentialSmoothing(
        train,
        trend="add",
        damped_trend=True,
        initialization_method="estimated",
    ).fit(optimized=True)
    return np.asarray(fit.forecast(h), dtype=float)


def _fc_holt_winters(train: pd.Series, h: int, seasonal: str) -> np.ndarray:
    fit = ExponentialSmoothing(
        train,
        trend="add",
        damped_trend=True,
        seasonal=seasonal,
        seasonal_periods=SEASON,
        initialization_method="estimated",
    ).fit(optimized=True)
    return np.asarray(fit.forecast(h), dtype=float)


@dataclass(frozen=True)
class _Candidate:
    key: str
    label: str
    fn: Callable[[pd.Series, int], np.ndarray]
    seasonal: bool = False


BASELINE_KEY = "seasonal_naive"
BASELINE_LABEL = "Même mois l'an dernier"


class ForecastingModel:
    DEFAULT_HORIZON = 6
    MAX_HORIZON = 120
    MIN_HISTORY_POINTS = 6
    MIN_SEASONAL_TRAIN = 2 * SEASON
    MAX_BACKTEST_HORIZON = 6
    # Trois fenêtres suffisent pour une validation glissante utile tout en
    # évitant 30+ ajustements statsmodels par requête.
    MAX_FOLDS = 3
    OUTLIER_Z = 5.0
    MODEL_NAME = "Sélection automatique"

    def __init__(
        self,
        date_column: str = "Date",
        value_column: str = "Chiffre_Affaires",
        non_negative: bool = True,
    ) -> None:
        self.date_column = date_column
        self.value_column = value_column
        self.non_negative = non_negative

    def run(
        self,
        dataframe: pd.DataFrame,
        horizon: int | None = None,
    ) -> ForecastResult:
        horizon = self._validate_horizon(horizon)
        prepared = self._prepare_dataframe(dataframe)
        if len(prepared) < self.MIN_HISTORY_POINTS:
            raise ValueError(
                f"Le forecasting nécessite au minimum {self.MIN_HISTORY_POINTS} périodes historiques."
            )

        series, missing_filled = self._build_monthly_series(prepared)
        if len(series) < self.MIN_HISTORY_POINTS:
            raise ValueError("Pas assez de données historiques après agrégation mensuelle pour effectuer une prévision.")

        quality_warnings, has_outliers = self._quality_checks(series, missing_filled)
        selection = self._select_model(series, horizon)
        chosen = selection["chosen"]
        values = self._predict(chosen, series, horizon)
        future_dates = _future_index(series, horizon)

        forecast = pd.DataFrame({self.date_column: future_dates, self.value_column: values})
        lower, upper = self._build_interval(values, selection["step_sigma"])
        forecast["Lower"] = lower
        forecast["Upper"] = upper

        historical = series.reset_index()
        historical.columns = [self.date_column, self.value_column]

        reliability = self._reliability(
            selection["metrics"].get("MAPE"), len(series), has_outliers
        )

        return ForecastResult(
            historical=historical,
            forecast=forecast,
            metrics=selection["metrics"],
            horizon_months=horizon,
            model_name=chosen.label,
            date_column=self.date_column,
            value_column=self.value_column,
            reliability=reliability,
            selection_reason=selection["reason"],
            baseline_name=BASELINE_LABEL,
            baseline_metrics=selection["baseline_metrics"],
            candidates=selection["ranking"],
            quality_warnings=quality_warnings,
            backtest_folds=selection["n_folds"],
            backtest_horizon=selection["test_size"],
        )

    def _candidates(self, series: pd.Series) -> list[_Candidate]:
        candidates = [
            _Candidate(
                key="seasonal_naive",
                label="Même mois l'an dernier",
                fn=_fc_seasonal_naive,
                seasonal=True,
            ),
            _Candidate(
                key="linear",
                label="Régression linéaire",
                fn=_fc_linear,
            ),
            _Candidate(
                key="holt_damped",
                label="Holt amorti",
                fn=_fc_holt_damped,
            ),
            _Candidate(
                key="linear_seasonal",
                label="Régression linéaire saisonnière",
                fn=_fc_linear_seasonal,
                seasonal=True,
            ),
            _Candidate(
                key="holt_winters",
                label="Holt-Winters",
                fn=lambda train, h: _fc_holt_winters(
                    train,
                    h,
                    seasonal="add",
                ),
                seasonal=True,
            ),
        ]

        # --------------------------------------------------------------
        # XGBOOST
        # --------------------------------------------------------------
        #
        # XGBoost utilise notamment lag_12.
        # On évite donc de l'activer sur des séries trop courtes.
        #
        if XGBOOST_AVAILABLE and len(series) >= 18:
            candidates.append(
                _Candidate(
                    key="xgboost",
                    label="XGBoost",
                    fn=forecast_xgboost,
                    seasonal=False,
                )
            )

        return candidates

    @staticmethod
    def _origins(n: int, test_size: int, min_train: int, max_folds: int) -> list[int]:
        last = n - test_size
        if last < min_train:
            return []
        step = max(1, (last - min_train) // max(1, max_folds - 1))
        return sorted(list(range(last, min_train - 1, -step))[:max_folds])

    def _select_model(self, series: pd.Series, horizon: int) -> dict:
        n = len(series)
        test_size = min(
            horizon,
            self.MAX_BACKTEST_HORIZON,
            max(1, n // 4),
        )

        all_candidates = self._candidates(series)

        seasonal_origins = self._origins(
            n,
            test_size,
            self.MIN_SEASONAL_TRAIN,
            self.MAX_FOLDS,
        )

        if len(seasonal_origins) >= 2:
            origins = seasonal_origins
            candidates = all_candidates
        else:
            origins = self._origins(
                n,
                test_size,
                self.MIN_HISTORY_POINTS,
                self.MAX_FOLDS,
            )

            candidates = [
                c for c in all_candidates
                if not c.seasonal
            ]

        if not origins:
            return self._unvalidated_selection(
                all_candidates,
                test_size,
            )

        outcomes: dict[
            str,
            tuple[np.ndarray, np.ndarray]
        ] = {}

        for candidate in candidates:
            actual_rows: list[np.ndarray] = []
            pred_rows: list[np.ndarray] = []
            failed = False

            for origin in origins:
                train = series.iloc[:origin]
                test = series.iloc[
                    origin: origin + test_size
                ]

                try:
                    pred = self._predict(
                        candidate,
                        train,
                        test_size,
                    )

                except Exception:
                    logger.warning(
                        "Modèle %s ignoré "
                        "(échec au backtest).",
                        candidate.key,
                        exc_info=True,
                    )
                    failed = True
                    break

                actual_rows.append(
                    test.to_numpy(dtype=float)
                )
                pred_rows.append(pred)

            if not failed:
                outcomes[candidate.key] = (
                    np.array(actual_rows),
                    np.array(pred_rows),
                )

        if not outcomes:
            return self._unvalidated_selection(
                all_candidates,
                test_size,
            )

        labels = {
            c.key: c
            for c in candidates
        }

        scored = {
            key: self._metrics(actual, pred)
            for key, (actual, pred)
            in outcomes.items()
        }

        ranking = [
            {
                "key": key,
                "model": labels[key].label,
                **metrics,
            }
            for key, metrics in sorted(
                scored.items(),
                key=lambda item: item[1]["MAE"],
            )
        ]

#..................................

        logger.info(
            "🏆 Classement des modèles de forecasting : %s",
            [
                {
                    "model": row["model"],
                    "MAE": round(row["MAE"], 2),
                    "RMSE": round(row["RMSE"], 2),
                    "MAPE": round(row["MAPE"], 4),
                }
                for row in ranking
            ],
        )        

#...................

        best_key = ranking[0]["key"]
        chosen = labels[best_key]

        actual, pred = outcomes[best_key]
        errors = actual - pred

        step_sigma = np.maximum.accumulate(
            np.sqrt(np.mean(errors ** 2, axis=0))
        )

        baseline_metrics = scored.get(BASELINE_KEY, {})

        logger.info(
            "🏆 Modèle sélectionné : %s | MAE=%.2f | RMSE=%.2f | MAPE=%.4f",
            chosen.label,
            scored[best_key]["MAE"],
            scored[best_key]["RMSE"],
            scored[best_key]["MAPE"],
        )

        return {
            "chosen": chosen,
            "metrics": scored[best_key],
            "baseline_metrics": baseline_metrics,
            "ranking": ranking,
            "step_sigma": step_sigma,
            "n_folds": len(origins),
            "test_size": test_size,
            "reason": self._explain(
                chosen,
                scored[best_key],
                baseline_metrics,
                len(origins),
                test_size,
            ),
        }


    

    @staticmethod
    def _unvalidated_selection(candidates: list[_Candidate], test_size: int) -> dict:
        chosen = next(c for c in candidates if c.key == "linear")
        return {
            "chosen": chosen,
            "metrics": {},
            "baseline_metrics": {},
            "ranking": [],
            "step_sigma": None,
            "n_folds": 0,
            "test_size": test_size,
            "reason": "Historique trop court pour valider un modèle : tendance linéaire appliquée sans validation.",
        }

    @staticmethod
    def _explain(chosen: _Candidate, metrics: dict[str, float], baseline_metrics: dict[str, float], n_folds: int, test_size: int) -> str:
        mae = f"{metrics['MAE']:,.0f}".replace(",", "\u00a0")
        text = f"{chosen.label} — erreur moyenne de {mae} par mois, validée sur {n_folds} fenêtres de {test_size} mois."
        if chosen.key == BASELINE_KEY:
            text += f" Aucun modèle plus élaboré ne fait mieux que la référence « {BASELINE_LABEL.lower()} »."
        elif baseline_metrics and baseline_metrics.get("MAE", 0) > 0:
            gain = (baseline_metrics["MAE"] - metrics["MAE"]) / baseline_metrics["MAE"] * 100
            text += f" Erreur {gain:.0f} % plus faible que la référence « {BASELINE_LABEL.lower()} »."
        return text

    def _predict(self, candidate: _Candidate, train: pd.Series, h: int) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = np.asarray(candidate.fn(train, h), dtype=float)
        if len(values) != h or not np.all(np.isfinite(values)):
            raise ValueError(f"Prévision invalide pour le modèle {candidate.key}.")
        return np.clip(values, 0, None) if self.non_negative else values

    def _build_interval(self, values: np.ndarray, step_sigma: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        h = len(values)
        if step_sigma is None or len(step_sigma) == 0:
            nan = np.full(h, np.nan)
            return nan, nan.copy()
        tested = len(step_sigma)
        sigma = np.array([
            step_sigma[j] if j < tested else step_sigma[-1] * np.sqrt((j + 1) / tested)
            for j in range(h)
        ])
        lower = values - Z_80 * sigma
        upper = values + Z_80 * sigma
        if self.non_negative:
            lower = np.clip(lower, 0, None)
        return lower, upper

    def _reliability(self, mape: float | None, n_points: int, has_outliers: bool) -> str:
        if mape is None:
            return "faible"
        level = 2 if mape <= 10 else 1 if mape <= 20 else 0
        if n_points < self.MIN_SEASONAL_TRAIN:
            level = min(level, 1)
        if has_outliers:
            level = min(level, 1)
        return RELIABILITY_LEVELS[level]

    @staticmethod
    def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
        actual = np.asarray(actual, dtype=float).ravel()
        predicted = np.asarray(predicted, dtype=float).ravel()
        errors = actual - predicted
        mask = actual != 0
        mape = float(np.mean(np.abs(errors[mask] / actual[mask])) * 100) if np.any(mask) else 0.0
        return {
            "MAE": float(np.mean(np.abs(errors))),
            "RMSE": float(np.sqrt(np.mean(errors ** 2))),
            "MAPE": mape,
        }

    def _quality_checks(self, series: pd.Series, missing_filled: int) -> tuple[list[str], bool]:
        notes: list[str] = []
        if (series < 0).any():
            notes.append("L'historique contient des valeurs négatives (avoirs, retours ?).")
        if missing_filled:
            notes.append(f"{missing_filled} mois manquant(s) complété(s) à 0.")
        outliers = self._detect_outliers(series)
        if outliers:
            months = ", ".join(d.strftime("%m/%Y") for d in outliers[:5])
            notes.append(f"Valeurs exceptionnelles détectées dans l'historique ({months}) : la prévision peut être faussée.")
        current_month = pd.Timestamp.today().normalize().replace(day=1)
        last_month = series.index.max()
        if last_month > current_month:
            notes.append("L'historique contient des mois postérieurs à aujourd'hui (données prévisionnelles ?).")
        elif last_month == current_month:
            notes.append("Le dernier mois de l'historique est le mois en cours : il peut être incomplet.")
        return notes, bool(outliers)

    def _detect_outliers(self, series: pd.Series) -> list[pd.Timestamp]:
        if len(series) < 8:
            return []
        rolling = series.rolling(7, center=True, min_periods=3).median()
        residual = series - rolling
        centre = residual.median()
        mad = float(np.median(np.abs(residual - centre)))
        if mad == 0:
            return []
        z = 0.6745 * (residual - centre) / mad
        return list(series.index[np.abs(z) > self.OUTLIER_Z])

    def _prepare_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if dataframe is None:
            raise ValueError("Le DataFrame fourni au forecasting est None.")
        if dataframe.empty:
            raise ValueError("Le DataFrame fourni au forecasting est vide.")
        df = dataframe.copy()
        date_column = self._detect_date_column(df)
        if date_column is None:
            raise ValueError("Impossible d'identifier une colonne temporelle.")
        value_column = self._detect_value_column(df, exclude=[date_column])
        if value_column is None:
            raise ValueError("Impossible d'identifier la colonne numérique à prévoir.")
        self.date_column = date_column
        self.value_column = value_column
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
        return df.dropna(subset=[date_column, value_column]).sort_values(date_column)

    def _build_monthly_series(self, dataframe: pd.DataFrame) -> tuple[pd.Series, int]:
        df = dataframe.copy()
        df["_month"] = df[self.date_column].dt.to_period("M").dt.to_timestamp()
        monthly = df.groupby("_month")[self.value_column].sum().sort_index()
        full_index = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
        missing = len(full_index) - len(monthly)
        monthly = monthly.reindex(full_index, fill_value=0)
        monthly.index.name = self.date_column
        return monthly.astype(float), missing

    @classmethod
    def _validate_horizon(cls, horizon: int | None) -> int:
        if horizon is None:
            return cls.DEFAULT_HORIZON
        try:
            horizon = int(horizon)
        except (TypeError, ValueError):
            return cls.DEFAULT_HORIZON
        if horizon <= 0:
            return cls.DEFAULT_HORIZON
        if horizon > cls.MAX_HORIZON:
            raise ValueError(f"L'horizon maximum autorisé pour le forecasting est de {cls.MAX_HORIZON} mois.")
        return horizon

    @staticmethod
    def _detect_date_column(dataframe: pd.DataFrame) -> str | None:
        preferred_names = [
            "Date", "date", "Date_Vente", "date_vente", "Date_Commande", "date_commande",
            "Mois", "mois", "Month", "month", "Annee_Mois", "annee_mois",
        ]
        for column in preferred_names:
            if column in dataframe.columns:
                return column
        for column in dataframe.columns:
            if pd.api.types.is_datetime64_any_dtype(dataframe[column]):
                return column
        for column in dataframe.columns:
            if pd.api.types.is_numeric_dtype(dataframe[column]):
                continue
            converted = pd.to_datetime(dataframe[column], errors="coerce")
            if converted.notna().mean() >= 0.8:
                return column
        return None

    @staticmethod
    def _detect_value_column(dataframe: pd.DataFrame, exclude: list[str] | None = None) -> str | None:
        exclude = exclude or []
        preferred_names = [
            "Chiffre_Affaires", "chiffre_affaires", "CA", "ca", "Ventes", "ventes",
            "Montant", "montant", "Revenue", "revenue", "Sales", "sales", "Total_Ventes", "total_ventes",
        ]
        for column in preferred_names:
            if column in dataframe.columns and column not in exclude:
                return column
        numeric_columns = [c for c in dataframe.select_dtypes(include="number").columns if c not in exclude]
        if numeric_columns:
            return numeric_columns[0]
        for column in dataframe.columns:
            if column in exclude:
                continue
            converted = pd.to_numeric(dataframe[column], errors="coerce")
            if converted.notna().mean() >= 0.8:
                return column
        return None


def run_forecasting(dataframe: pd.DataFrame, horizon: int = 6) -> ForecastResult:
    return ForecastingModel().run(dataframe=dataframe, horizon=horizon)

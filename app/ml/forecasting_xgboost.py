


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import logging
import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False


logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

DEFAULT_LAGS = (1, 2, 3, 4, 5, 6, 9, 12)

# Nombre minimal de points nécessaires pour construire
# correctement les features avec lag/rolling/growth sur 12 mois.
MIN_TRAIN_POINTS = 15

MAX_BACKTEST_FOLDS = 3


@dataclass(frozen=True)
class XGBoostConfig:
    """Configuration d'un modèle XGBoost."""

    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: int
    reg_alpha: float
    reg_lambda: float


# Petit ensemble de configurations volontairement compact.
# L'objectif est de trouver une configuration robuste sans
# transformer l'optimisation en GridSearch trop coûteux.
XGBOOST_CONFIGS: Tuple[XGBoostConfig, ...] = (
    XGBoostConfig(
        n_estimators=400,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.0,
        reg_lambda=1.0,
    ),
    XGBoostConfig(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.0,
        reg_lambda=1.0,
    ),
    XGBoostConfig(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.0,
        reg_lambda=1.5,
    ),
    XGBoostConfig(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.5,
    ),
    XGBoostConfig(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=2.0,
    ),
)


class XGBoostForecaster:
    """
    Prévision temporelle avec XGBoost.

    Fonctionnement :
    1. construction des features temporelles ;
    2. optimisation interne de la configuration XGBoost ;
    3. entraînement final sur toute l'historique ;
    4. prévision récursive mois par mois.
    """

    def __init__(
        self,
        random_state: int = 42,
        optimize: bool = True,
    ) -> None:
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost n'est pas installé. "
                "Installez-le avec : pip install xgboost"
            )

        self.random_state = random_state
        self.optimize = optimize

        self.model = None
        self.best_config: XGBoostConfig | None = None
        self.feature_columns: List[str] = []

    # ========================================================
    # Validation
    # ========================================================

    @staticmethod
    def _validate_series(series: pd.Series) -> pd.Series:
        """Valide et normalise la série temporelle."""

        if not isinstance(series, pd.Series):
            raise TypeError("series doit être une pandas.Series.")

        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError(
                "L'index de la série doit être un pandas.DatetimeIndex."
            )

        result = series.copy()

        result = pd.to_numeric(result, errors="coerce")

        if result.isna().any():
            raise ValueError(
                "La série contient des valeurs non numériques ou manquantes."
            )

        if not np.all(np.isfinite(result.values)):
            raise ValueError(
                "La série contient des valeurs infinies ou non finies."
            )

        result = result.sort_index()

        return result.astype(float)

    # ========================================================
    # Feature engineering
    # ========================================================

    def _build_features(self, series: pd.Series) -> pd.DataFrame:
        """
        Construit les variables explicatives à partir de l'historique.
        """

        df = pd.DataFrame(index=series.index)

        # ----------------------------------------------------
        # Lags
        # ----------------------------------------------------

        for lag in DEFAULT_LAGS:
            df[f"lag_{lag}"] = series.shift(lag)

        # Lag 24 uniquement lorsque l'historique est suffisamment long.
        if len(series) >= 30:
            df["lag_24"] = series.shift(24)

        # ----------------------------------------------------
        # Rolling statistics
        # ----------------------------------------------------

        df["rolling_mean_3"] = series.shift(1).rolling(3).mean()
        df["rolling_mean_6"] = series.shift(1).rolling(6).mean()
        df["rolling_mean_12"] = series.shift(1).rolling(12).mean()

        df["rolling_std_3"] = series.shift(1).rolling(3).std()
        df["rolling_std_12"] = series.shift(1).rolling(12).std()

        # ----------------------------------------------------
        # Growth / différences
        # ----------------------------------------------------

        df["growth_1"] = series.shift(1).pct_change()
        df["growth_12"] = series.shift(1).pct_change(12)

        df["diff_1"] = series.shift(1).diff(1)
        df["diff_12"] = series.shift(1).diff(12)

        # ----------------------------------------------------
        # Variables calendaires
        # ----------------------------------------------------

        month = series.index.month
        quarter = series.index.quarter
        year = series.index.year

        df["month"] = month
        df["quarter"] = quarter
        df["year"] = year

        # Encodage cyclique du mois
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)

        # Tendance globale
        df["trend"] = np.arange(len(series), dtype=float)

        return df

    # ========================================================
    # Création du modèle
    # ========================================================

    def _create_model(self, config: XGBoostConfig):
        """Crée une instance XGBRegressor."""

        return XGBRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            min_child_weight=config.min_child_weight,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=-1,
        )

    # ========================================================
    # Backtesting interne
    # ========================================================

    def _get_backtest_folds(
        self,
        n: int,
    ) -> List[Tuple[int, int]]:
        """
        Construit les folds temporels.

        Chaque tuple contient :

            (origin, test_size)

        Exemple avec une série suffisamment longue :

            fold 1 : historique -> validation
            fold 2 : historique étendu -> validation
            fold 3 : historique encore étendu -> validation

        On utilise une fenêtre de validation adaptée à la taille
        de la série afin d'éviter des folds trop petits.
        """

        if n < MIN_TRAIN_POINTS + 1:
            return []

        # ----------------------------------------------------
        # Taille de validation adaptée à l'historique
        # ----------------------------------------------------

        if n >= 30:
            test_size = 3
        elif n >= 24:
            test_size = 2
        else:
            test_size = 1

        # On construit des origines espacées de test_size.
        #
        # Exemple n=30, test_size=3 :
        #
        # origin 21 -> [21,22,23]
        # origin 24 -> [24,25,26]
        # origin 27 -> [27,28,29]
        #
        # Le dernier fold teste donc les dernières observations.

        first_origin = max(
            MIN_TRAIN_POINTS,
            n - MAX_BACKTEST_FOLDS * test_size,
        )

        last_origin = n - test_size

        origins = list(
            range(
                first_origin,
                last_origin + 1,
                test_size,
            )
        )

        # On garde uniquement les derniers folds si nécessaire.
        if len(origins) > MAX_BACKTEST_FOLDS:
            origins = origins[-MAX_BACKTEST_FOLDS:]

        folds = [
            (origin, test_size)
            for origin in origins
            if origin >= MIN_TRAIN_POINTS
            and origin + test_size <= n
        ]

        return folds

    # ========================================================

    @staticmethod
    def _mae(
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> float:
        """Calcule la MAE."""

        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)

        return float(np.mean(np.abs(actual - predicted)))

    # ========================================================

    def _recursive_predict_with_model(
        self,
        model,
        history: pd.Series,
        horizon: int,
    ) -> np.ndarray:
        """
        Prévision récursive.

        À chaque mois, la prédiction précédente est ajoutée à
        l'historique afin de construire les features du mois suivant.
        """

        history = history.copy().astype(float)

        predictions: List[float] = []

        for _ in range(horizon):
            next_features = self._build_next_features(
                history,
                model,
            )

            prediction = float(model.predict(next_features)[0])

            if not np.isfinite(prediction):
                raise ValueError(
                    "XGBoost a produit une prédiction non finie."
                )

            # Les indicateurs financiers prévus ne doivent normalement
            # pas devenir négatifs.
            prediction = max(0.0, prediction)

            predictions.append(prediction)

            next_date = history.index[-1] + pd.offsets.MonthBegin(1)

            history.loc[next_date] = prediction

        return np.asarray(predictions, dtype=float)

    # ========================================================

    def _evaluate_config(
        self,
        series: pd.Series,
        config: XGBoostConfig,
        folds: List[Tuple[int, int]],
    ) -> Tuple[float, List[float]]:
        """
        Évalue une configuration sur tous les folds temporels.

        Retourne :

            moyenne MAE
            liste des MAE par fold
        """

        fold_scores: List[float] = []

        for fold_number, (origin, test_size) in enumerate(
            folds,
            start=1,
        ):
            train = series.iloc[:origin]
            validation = series.iloc[
                origin : origin + test_size
            ]

            try:
                features = self._build_features(train)

                dataset = features.copy()
                dataset["target"] = train.values

                dataset = dataset.dropna()

                if dataset.empty:
                    logger.debug(
                        "XGBoost config ignorée sur fold %d : "
                        "aucune ligne exploitable.",
                        fold_number,
                    )
                    continue

                X_train = dataset.drop(columns=["target"])
                y_train = dataset["target"]

                if len(X_train) < 2:
                    continue

                model = self._create_model(config)

                model.fit(
                    X_train,
                    y_train,
                    verbose=False,
                )

                predictions = self._recursive_predict_with_model(
                    model=model,
                    history=train,
                    horizon=len(validation),
                )

                score = self._mae(
                    validation.values,
                    predictions,
                )

                if np.isfinite(score):
                    fold_scores.append(score)

            except Exception as exc:
                logger.debug(
                    "Erreur XGBoost config sur fold %d : %s",
                    fold_number,
                    exc,
                )

        if not fold_scores:
            return float("inf"), []

        return float(np.mean(fold_scores)), fold_scores

    # ========================================================

    def _select_best_config(
        self,
        series: pd.Series,
    ) -> XGBoostConfig:
        """
        Sélectionne la meilleure configuration XGBoost via
        backtesting temporel multi-fold.

        La sélection est faite sur la MAE moyenne des folds.
        """

        folds = self._get_backtest_folds(len(series))

        # Si la série est trop courte pour un vrai backtesting,
        # on utilise une configuration robuste par défaut.
        if not folds:
            logger.info(
                "XGBoost : historique trop court pour le "
                "backtesting multi-fold. Configuration par défaut."
            )

            return XGBOOST_CONFIGS[1]

        logger.info(
            "XGBoost : optimisation sur %d fold(s) temporel(s).",
            len(folds),
        )

        results = []

        for config_number, config in enumerate(
            XGBOOST_CONFIGS,
            start=1,
        ):
            mean_mae, fold_scores = self._evaluate_config(
                series=series,
                config=config,
                folds=folds,
            )

            results.append(
                {
                    "config_number": config_number,
                    "config": config,
                    "mean_mae": mean_mae,
                    "fold_scores": fold_scores,
                }
            )

            if np.isfinite(mean_mae):
                logger.info(
                    (
                        "XGBoost config %d/%d | "
                        "MAE moyenne=%.2f | "
                        "folds=%s"
                    ),
                    config_number,
                    len(XGBOOST_CONFIGS),
                    mean_mae,
                    [
                        round(score, 2)
                        for score in fold_scores
                    ],
                )
            else:
                logger.info(
                    "XGBoost config %d/%d | échec",
                    config_number,
                    len(XGBOOST_CONFIGS),
                )

        valid_results = [
            result
            for result in results
            if np.isfinite(result["mean_mae"])
        ]

        if not valid_results:
            logger.warning(
                "Aucune configuration XGBoost valide. "
                "Utilisation de la configuration par défaut."
            )

            return XGBOOST_CONFIGS[1]

        best = min(
            valid_results,
            key=lambda item: item["mean_mae"],
        )

        logger.info(
            (
                "🏆 XGBoost configuration sélectionnée : "
                "config %d | MAE moyenne=%.2f | "
                "scores=%s"
            ),
            best["config_number"],
            best["mean_mae"],
            [
                round(score, 2)
                for score in best["fold_scores"]
            ],
        )

        return best["config"]

    # ========================================================
    # Fit
    # ========================================================

    def fit(
        self,
        series: pd.Series,
    ) -> "XGBoostForecaster":
        """
        Entraîne XGBoost sur toute la série historique.
        """

        series = self._validate_series(series)

        if len(series) < MIN_TRAIN_POINTS:
            raise ValueError(
                f"XGBoost nécessite au moins "
                f"{MIN_TRAIN_POINTS} observations."
            )

        # ----------------------------------------------------
        # Optimisation de la configuration
        # ----------------------------------------------------

        if self.optimize:
            self.best_config = self._select_best_config(series)
        else:
            self.best_config = XGBOOST_CONFIGS[1]

        logger.info(
            "XGBoost : entraînement final avec config=%s",
            self.best_config,
        )

        # ----------------------------------------------------
        # Features finales
        # ----------------------------------------------------

        features = self._build_features(series)

        dataset = features.copy()
        dataset["target"] = series.values

        dataset = dataset.dropna()

        if dataset.empty:
            raise ValueError(
                "Impossible de construire les features XGBoost."
            )

        X_train = dataset.drop(columns=["target"])
        y_train = dataset["target"]

        self.feature_columns = list(X_train.columns)

        # ----------------------------------------------------
        # Modèle final
        # ----------------------------------------------------

        self.model = self._create_model(
            self.best_config
        )

        self.model.fit(
            X_train,
            y_train,
            verbose=False,
        )

        return self

    # ========================================================
    # Features pour le prochain mois
    # ========================================================

    def _build_next_features(
        self,
        history: pd.Series,
        model,
    ) -> pd.DataFrame:
        """
        Construit les features du prochain mois.

        Les colonnes sont alignées exactement sur celles utilisées
        lors de l'entraînement du modèle.
        """

        next_date = (
            history.index[-1]
            + pd.offsets.MonthBegin(1)
        )

        row: Dict[str, float] = {}

        # ----------------------------------------------------
        # Lags
        # ----------------------------------------------------

        for lag in DEFAULT_LAGS:
            if len(history) >= lag:
                row[f"lag_{lag}"] = float(
                    history.iloc[-lag]
                )
            else:
                row[f"lag_{lag}"] = np.nan

        # Lag 24 si le modèle final l'utilise.
        if "lag_24" in model.feature_names_in_:
            if len(history) >= 24:
                row["lag_24"] = float(
                    history.iloc[-24]
                )
            else:
                row["lag_24"] = np.nan

        # ----------------------------------------------------
        # Rolling
        # ----------------------------------------------------

        shifted = history.shift(1)

        row["rolling_mean_3"] = float(
            shifted.tail(3).mean()
        )

        row["rolling_mean_6"] = float(
            shifted.tail(6).mean()
        )

        row["rolling_mean_12"] = float(
            shifted.tail(12).mean()
        )

        row["rolling_std_3"] = float(
            shifted.tail(3).std()
        )

        row["rolling_std_12"] = float(
            shifted.tail(12).std()
        )

        # ----------------------------------------------------
        # Growth
        # ----------------------------------------------------

        if len(history) >= 2:
            previous = float(history.iloc[-1])
            before = float(history.iloc[-2])

            if before != 0:
                row["growth_1"] = (
                    previous / before
                ) - 1.0
            else:
                row["growth_1"] = 0.0
        else:
            row["growth_1"] = 0.0

        if len(history) >= 13:
            current = float(history.iloc[-1])
            previous_year = float(history.iloc[-13])

            if previous_year != 0:
                row["growth_12"] = (
                    current / previous_year
                ) - 1.0
            else:
                row["growth_12"] = 0.0
        else:
            row["growth_12"] = 0.0

        # ----------------------------------------------------
        # Différences
        # ----------------------------------------------------

        if len(history) >= 2:
            row["diff_1"] = float(
                history.iloc[-1]
                - history.iloc[-2]
            )
        else:
            row["diff_1"] = 0.0

        if len(history) >= 13:
            row["diff_12"] = float(
                history.iloc[-1]
                - history.iloc[-13]
            )
        else:
            row["diff_12"] = 0.0

        # ----------------------------------------------------
        # Calendrier
        # ----------------------------------------------------

        month = next_date.month
        quarter = next_date.quarter
        year = next_date.year

        row["month"] = month
        row["quarter"] = quarter
        row["year"] = year

        row["month_sin"] = np.sin(
            2 * np.pi * month / 12
        )

        row["month_cos"] = np.cos(
            2 * np.pi * month / 12
        )

        # ----------------------------------------------------
        # Tendance
        # ----------------------------------------------------

        row["trend"] = float(len(history))

        features = pd.DataFrame(
            [row],
            index=[next_date],
        )

        # ----------------------------------------------------
        # Alignement strict avec les features d'entraînement
        # ----------------------------------------------------

        expected_columns = list(
            model.feature_names_in_
        )

        features = features.reindex(
            columns=expected_columns
        )

        # Certaines features peuvent être NaN avec un historique
        # court. On tente ici de les remplacer de manière robuste.
        features = features.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        features = features.fillna(0.0)

        return features

    # ========================================================
    # Predict
    # ========================================================

    def predict(
        self,
        series: pd.Series,
        horizon: int,
    ) -> np.ndarray:
        """
        Prévoit les prochains mois.
        """

        if self.model is None:
            raise RuntimeError(
                "Le modèle XGBoost doit être entraîné "
                "avant d'appeler predict()."
            )

        if horizon <= 0:
            raise ValueError(
                "horizon doit être strictement positif."
            )

        series = self._validate_series(series)

        return self._recursive_predict_with_model(
            model=self.model,
            history=series,
            horizon=horizon,
        )


# ============================================================
# Fonction publique utilisée par forecasting.py
# ============================================================

def forecast_xgboost(
    series: pd.Series,
    horizon: int,
) -> np.ndarray:
    """
    Interface publique de forecasting.py.

    Exemple :

        forecast = forecast_xgboost(series, 6)
    """

    if not XGBOOST_AVAILABLE:
        raise ImportError(
            "XGBoost n'est pas disponible."
        )

    forecaster = XGBoostForecaster(
        optimize=True,
    )

    forecaster.fit(series)

    return forecaster.predict(
        series,
        horizon,
    )

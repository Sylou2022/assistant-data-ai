
# Code : ml/evaluation.py

"""
app/ml/evaluation.py

Évaluation des modèles ML de l'Assistant Data IA.

Responsabilités :
- calcul du MAE
- calcul du RMSE
- calcul du MAPE
- calcul de plusieurs métriques simultanément
- comparaison des valeurs réelles et prédites
- génération d'un résultat d'évaluation standardisé

Ce module est indépendant du modèle utilisé.
Il pourra donc être utilisé par :
    - forecasting
    - anomaly detection
    - classification
    - autres modèles ML
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


@dataclass
class EvaluationResult:
    """
    Résultat standardisé de l'évaluation d'un modèle.
    """

    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None

    sample_count: int = 0

    success: bool = True
    error: Optional[str] = None


class ModelEvaluator:
    """
    Évaluateur générique pour les modèles ML.

    Les métriques principales sont :

        MAE
            Mean Absolute Error

        RMSE
            Root Mean Squared Error

        MAPE
            Mean Absolute Percentage Error
    """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        y_true,
        y_pred,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Valide et convertit les valeurs réelles et prédites.
        """

        if y_true is None:
            raise ValueError(
                "Les valeurs réelles (y_true) sont obligatoires."
            )

        if y_pred is None:
            raise ValueError(
                "Les valeurs prédites (y_pred) sont obligatoires."
            )

        true_values = np.asarray(y_true, dtype=float)
        predicted_values = np.asarray(y_pred, dtype=float)

        if true_values.size == 0:
            raise ValueError(
                "Les valeurs réelles sont vides."
            )

        if predicted_values.size == 0:
            raise ValueError(
                "Les valeurs prédites sont vides."
            )

        if true_values.shape != predicted_values.shape:
            raise ValueError(
                "y_true et y_pred doivent avoir la même dimension."
            )

        if not np.all(np.isfinite(true_values)):
            raise ValueError(
                "y_true contient des valeurs invalides."
            )

        if not np.all(np.isfinite(predicted_values)):
            raise ValueError(
                "y_pred contient des valeurs invalides."
            )

        return true_values, predicted_values

    # ------------------------------------------------------------------
    # MAE
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_mae(
        y_true,
        y_pred,
    ) -> float:
        """
        Calcule le Mean Absolute Error.

        MAE = moyenne(|réel - prédit|)
        """

        true_values, predicted_values = (
            ModelEvaluator._validate_inputs(
                y_true,
                y_pred,
            )
        )

        return float(
            mean_absolute_error(
                true_values,
                predicted_values,
            )
        )

    # ------------------------------------------------------------------
    # RMSE
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_rmse(
        y_true,
        y_pred,
    ) -> float:
        """
        Calcule le Root Mean Squared Error.

        RMSE = sqrt(moyenne((réel - prédit)^2))
        """

        true_values, predicted_values = (
            ModelEvaluator._validate_inputs(
                y_true,
                y_pred,
            )
        )

        mse = mean_squared_error(
            true_values,
            predicted_values,
        )

        return float(np.sqrt(mse))

    # ------------------------------------------------------------------
    # MAPE
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_mape(
        y_true,
        y_pred,
    ) -> float:
        """
        Calcule le Mean Absolute Percentage Error.

        Les valeurs réelles égales à zéro sont ignorées
        afin d'éviter une division par zéro.

        Le résultat est exprimé en pourcentage.
        """

        true_values, predicted_values = (
            ModelEvaluator._validate_inputs(
                y_true,
                y_pred,
            )
        )

        non_zero_mask = true_values != 0

        if not np.any(non_zero_mask):
            raise ValueError(
                "Impossible de calculer le MAPE : "
                "toutes les valeurs réelles sont égales à zéro."
            )

        filtered_true = true_values[non_zero_mask]
        filtered_predicted = predicted_values[non_zero_mask]

        mape = np.mean(
            np.abs(
                (filtered_true - filtered_predicted)
                / filtered_true
            )
        ) * 100

        return float(mape)

    # ------------------------------------------------------------------
    # Toutes les métriques
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true,
        y_pred,
    ) -> EvaluationResult:
        """
        Calcule l'ensemble des métriques disponibles.
        """

        try:
            true_values, predicted_values = (
                self._validate_inputs(
                    y_true,
                    y_pred,
                )
            )

            mae = self.calculate_mae(
                true_values,
                predicted_values,
            )

            rmse = self.calculate_rmse(
                true_values,
                predicted_values,
            )

            try:
                mape = self.calculate_mape(
                    true_values,
                    predicted_values,
                )
            except ValueError:
                mape = None

            return EvaluationResult(
                mae=mae,
                rmse=rmse,
                mape=mape,
                sample_count=len(true_values),
                success=True,
            )

        except Exception as exc:
            return EvaluationResult(
                success=False,
                error=str(exc),
            )


# ----------------------------------------------------------------------
# Fonction pratique
# ----------------------------------------------------------------------

def evaluate_predictions(
    y_true,
    y_pred,
) -> EvaluationResult:
    """
    Fonction simplifiée pour évaluer directement des prédictions.

    Exemple :

        result = evaluate_predictions(
            y_true,
            y_pred,
        )

        print(result.mae)
        print(result.rmse)
        print(result.mape)
    """

    evaluator = ModelEvaluator()

    return evaluator.evaluate(
        y_true,
        y_pred,
    )


# ----------------------------------------------------------------------
# Évaluation depuis un DataFrame
# ----------------------------------------------------------------------

def evaluate_dataframe(
    df: pd.DataFrame,
    actual_column: str,
    predicted_column: str,
) -> EvaluationResult:
    """
    Évalue un DataFrame contenant les valeurs réelles
    et les valeurs prédites.

    Exemple :

        df = pd.DataFrame({
            "Actual": [100, 120, 140],
            "Prediction": [105, 118, 137],
        })

        result = evaluate_dataframe(
            df,
            actual_column="Actual",
            predicted_column="Prediction",
        )
    """

    if df is None:
        return EvaluationResult(
            success=False,
            error="Le DataFrame fourni est None.",
        )

    if not isinstance(df, pd.DataFrame):
        return EvaluationResult(
            success=False,
            error="L'entrée doit être un pandas.DataFrame.",
        )

    missing_columns = {
        actual_column,
        predicted_column,
    } - set(df.columns)

    if missing_columns:
        return EvaluationResult(
            success=False,
            error=(
                "Colonnes manquantes : "
                f"{sorted(missing_columns)}"
            ),
        )

    clean_df = df[
        [actual_column, predicted_column]
    ].dropna()

    if clean_df.empty:
        return EvaluationResult(
            success=False,
            error="Aucune donnée valide à évaluer.",
        )

    return evaluate_predictions(
        clean_df[actual_column].values,
        clean_df[predicted_column].values,
    )


# ----------------------------------------------------------------------
# Test local
# ----------------------------------------------------------------------

if __name__ == "__main__":

    y_true = np.array([
        100,
        120,
        140,
        160,
        180,
    ])

    y_pred = np.array([
        105,
        118,
        137,
        165,
        175,
    ])

    print("=== ÉVALUATION DU MODÈLE ===")

    result = evaluate_predictions(
        y_true,
        y_pred,
    )

    if result.success:

        print(f"Nombre d'observations : {result.sample_count}")
        print(f"MAE                  : {result.mae:.2f}")
        print(f"RMSE                 : {result.rmse:.2f}")

        if result.mape is not None:
            print(f"MAPE                 : {result.mape:.2f}%")
        else:
            print("MAPE                 : non disponible")

    else:
        print(f"Erreur : {result.error}")



"""
app/ml/preprocessing.py

Préparation et nettoyage des données pour les modèles ML
de l'Assistant Data IA.

Responsabilités :
- validation des colonnes
- conversion des dates
- conversion des valeurs numériques
- gestion des valeurs manquantes
- agrégation temporelle
- tri chronologique
- contrôle du volume de données
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class PreprocessingConfig:
    """
    Configuration générale du preprocessing.
    """

    date_column: str = "Date"
    value_column: str = "Chiffre_Affaires"

    frequency: str = "MS"

    fill_missing: bool = True
    fill_method: str = "interpolate"

    min_periods: int = 3


class DataPreprocessor:
    """
    Prépare un DataFrame pour les traitements ML.

    Cette classe est volontairement indépendante du modèle.
    Elle peut donc être utilisée par :
        - forecasting
        - détection d'anomalies
        - clustering
        - classification
    """

    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
    ):
        self.config = config or PreprocessingConfig()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_columns(self, df: pd.DataFrame) -> None:
        """
        Vérifie que les colonnes nécessaires sont présentes.
        """

        if df is None:
            raise ValueError("Le DataFrame fourni est None.")

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Les données doivent être fournies sous forme de pandas.DataFrame."
            )

        required_columns = {
            self.config.date_column,
            self.config.value_column,
        }

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Colonnes manquantes : {sorted(missing_columns)}"
            )

    # ------------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------------

    def convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convertit la colonne de dates en datetime.
        """

        result = df.copy()

        result[self.config.date_column] = pd.to_datetime(
            result[self.config.date_column],
            errors="coerce",
        )

        return result

    # ------------------------------------------------------------------
    # Valeurs numériques
    # ------------------------------------------------------------------

    def convert_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convertit la colonne de valeurs en numérique.
        """

        result = df.copy()

        result[self.config.value_column] = pd.to_numeric(
            result[self.config.value_column],
            errors="coerce",
        )

        return result

    # ------------------------------------------------------------------
    # Nettoyage
    # ------------------------------------------------------------------

    def remove_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Supprime les lignes dont la date ou la valeur est invalide.
        """

        result = df.copy()

        result = result.dropna(
            subset=[
                self.config.date_column,
                self.config.value_column,
            ]
        )

        return result

    # ------------------------------------------------------------------
    # Tri chronologique
    # ------------------------------------------------------------------

    def sort_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trie les données par date.
        """

        result = df.copy()

        result = result.sort_values(
            by=self.config.date_column
        )

        result = result.reset_index(drop=True)

        return result

    # ------------------------------------------------------------------
    # Agrégation temporelle
    # ------------------------------------------------------------------

    def aggregate_time_series(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Agrège les valeurs selon la fréquence configurée.

        Exemple :
            frequency="MS"

        signifie que les données sont agrégées
        au début de chaque mois.

        Les valeurs sont additionnées.
        """

        result = df.copy()

        result = result.set_index(
            self.config.date_column
        )

        result = (
            result[self.config.value_column]
            .resample(self.config.frequency)
            .sum()
            .reset_index()
        )

        return result

    # ------------------------------------------------------------------
    # Gestion des valeurs manquantes
    # ------------------------------------------------------------------

    def fill_missing_values(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Gère les valeurs manquantes après agrégation temporelle.
        """

        result = df.copy()

        if not self.config.fill_missing:
            return result

        column = self.config.value_column

        if self.config.fill_method == "interpolate":
            result[column] = (
                result[column]
                .interpolate(method="linear")
                .ffill()
                .bfill()
            )

        elif self.config.fill_method == "ffill":
            result[column] = (
                result[column]
                .ffill()
                .bfill()
            )

        elif self.config.fill_method == "bfill":
            result[column] = (
                result[column]
                .bfill()
                .ffill()
            )

        elif self.config.fill_method == "zero":
            result[column] = result[column].fillna(0)

        else:
            raise ValueError(
                f"Méthode de remplissage inconnue : "
                f"{self.config.fill_method}"
            )

        return result

    # ------------------------------------------------------------------
    # Contrôle du volume
    # ------------------------------------------------------------------

    def validate_minimum_periods(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Vérifie que suffisamment de périodes sont disponibles
        pour entraîner un modèle ML.
        """

        number_of_periods = len(df)

        if number_of_periods < self.config.min_periods:
            raise ValueError(
                f"Pas assez de données pour le modèle ML. "
                f"Minimum requis : {self.config.min_periods} périodes, "
                f"données disponibles : {number_of_periods}."
            )

    # ------------------------------------------------------------------
    # Pipeline complet
    # ------------------------------------------------------------------

    def prepare_time_series(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Pipeline complet de préparation d'une série temporelle.

        Étapes :
            1. validation
            2. conversion des dates
            3. conversion numérique
            4. suppression des données invalides
            5. tri chronologique
            6. agrégation temporelle
            7. gestion des valeurs manquantes
            8. validation du nombre de périodes
        """

        self.validate_columns(df)

        result = self.convert_dates(df)

        result = self.convert_numeric(result)

        result = self.remove_invalid_rows(result)

        result = self.sort_by_date(result)

        result = self.aggregate_time_series(result)

        result = self.fill_missing_values(result)

        result = self.sort_by_date(result)

        self.validate_minimum_periods(result)

        return result


# ----------------------------------------------------------------------
# Fonction pratique
# ----------------------------------------------------------------------

def prepare_time_series(
    df: pd.DataFrame,
    date_column: str = "Date",
    value_column: str = "Chiffre_Affaires",
    frequency: str = "MS",
) -> pd.DataFrame:
    """
    Fonction simplifiée pour préparer une série temporelle.

    Exemple :

        df_prepared = prepare_time_series(
            df,
            date_column="Date",
            value_column="Chiffre_Affaires",
            frequency="MS",
        )
    """

    config = PreprocessingConfig(
        date_column=date_column,
        value_column=value_column,
        frequency=frequency,
    )

    preprocessor = DataPreprocessor(config)

    return preprocessor.prepare_time_series(df)


# ----------------------------------------------------------------------
# Test local
# ----------------------------------------------------------------------

if __name__ == "__main__":

    data = {
        "Date": [
            "2024-01-15",
            "2024-01-20",
            "2024-02-10",
            "2024-03-05",
            "2024-03-20",
            "2024-04-10",
        ],
        "Chiffre_Affaires": [
            12000,
            8000,
            15000,
            11000,
            9000,
            18000,
        ],
    }

    df = pd.DataFrame(data)

    print("=== DONNÉES ORIGINALES ===")
    print(df)

    prepared_df = prepare_time_series(
        df,
        date_column="Date",
        value_column="Chiffre_Affaires",
        frequency="MS",
    )

    print("\n=== DONNÉES PRÉPARÉES ===")
    print(prepared_df)

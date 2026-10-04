from __future__ import annotations

import pandas as pd


class ResultFormatter:
    def __init__(self, max_display_rows: int = 20) -> None:
        if max_display_rows <= 0:
            raise ValueError(
                "max_display_rows doit être supérieur à zéro."
            )

        self.max_display_rows = max_display_rows

    def format(self, df: pd.DataFrame) -> str:
        """
        Transforme un DataFrame en réponse lisible
        pour l'utilisateur.
        """

        if df.empty:
            return "Aucun résultat."

        lines: list[str] = []

        # Résumé technique
        lines.append(
            f"{len(df)} ligne(s) retournée(s)."
        )

        columns = ", ".join(
            str(column)
            for column in df.columns
        )

        lines.append(
            f"Colonnes : {columns}"
        )

        lines.append("")
        lines.append("Résultat :")
        lines.append("")

        # Tableau
        displayed_df = df.head(
            self.max_display_rows
        )

        lines.append(
            displayed_df.to_string(
                index=False
            )
        )

        # Information si le tableau est tronqué
        if len(df) > self.max_display_rows:
            lines.append("")
            lines.append(
                f"Affichage limité aux "
                f"{self.max_display_rows} premières lignes "
                f"sur {len(df)}."
            )

        return "\n".join(lines)
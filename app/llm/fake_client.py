from __future__ import annotations


from typing import Any


class FakeLLMClient:
    def generate_structured_response(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        user_message = messages[-1]["content"].lower()

        # ---------------------------------------------------------
        # CA par mois 2024
        # ---------------------------------------------------------

        if (
            "chiffre d'affaires" in user_message
            and "mois" in user_message
            and "2024" in user_message
        ) or (
            "ca par mois" in user_message
        ):
            return {
                "status": "ok",
                "sql": """
SELECT
    DATEFROMPARTS(
        YEAR(DateVente),
        MONTH(DateVente),
        1
    ) AS Mois,
    SUM(Montant) AS Chiffre_Affaires
FROM dbo.Fact_Ventes
WHERE DateVente >= '2024-01-01'
  AND DateVente < '2025-01-01'
GROUP BY
    YEAR(DateVente),
    MONTH(DateVente)
ORDER BY
    Mois;
""".strip(),
                "explanation": (
                    "Calcul du chiffre d'affaires mensuel pour 2024."
                ),
                "tables_used": [
                    "dbo.Fact_Ventes",
                ],
                "columns_used": [
                    "dbo.Fact_Ventes.DateVente",
                    "dbo.Fact_Ventes.Montant",
                ],
                "assumptions": [
                    "Le chiffre d'affaires correspond à la somme de Montant."
                ],
                "clarification_question": "",
            }

        # ---------------------------------------------------------
        # CA par région
        # ---------------------------------------------------------

        if (
            "chiffre d'affaires" in user_message
            and "région" in user_message
        ):
            return {
                "status": "ok",
                "sql": """
SELECT
    dr.Region,
    SUM(fv.Chiffre_Affaires)
        AS Chiffre_Affaires_Total
FROM dbo.Fact_Ventes AS fv
INNER JOIN dbo.Dim_Region AS dr
    ON fv.Region_ID = dr.Region_ID
GROUP BY
    dr.Region
ORDER BY
    Chiffre_Affaires_Total DESC;
""".strip(),
                "explanation": (
                    "Calcul du chiffre d'affaires total par région."
                ),
                "tables_used": [
                    "dbo.Fact_Ventes",
                    "dbo.Dim_Region",
                ],
                "columns_used": [
                    "dbo.Fact_Ventes.Chiffre_Affaires",
                    "dbo.Fact_Ventes.Region_ID",
                    "dbo.Dim_Region.Region_ID",
                    "dbo.Dim_Region.Region",
                ],
                "assumptions": [],
                "clarification_question": "",
            }

        # ---------------------------------------------------------
        # Marge par produit
        # ---------------------------------------------------------

        if (
            "marge" in user_message
            and "produit" in user_message
        ):
            return {
                "status": "ok",
                "sql": """
SELECT
    dp.Nom_Produit,
    SUM(fv.Marge) AS Marge_Totale
FROM dbo.Fact_Ventes AS fv
INNER JOIN dbo.Dim_Produit AS dp
    ON fv.Produit_ID = dp.Produit_ID
GROUP BY
    dp.Nom_Produit
ORDER BY
    Marge_Totale DESC;
""".strip(),
                "explanation": (
                    "Calcul de la marge totale par produit."
                ),
                "tables_used": [
                    "dbo.Fact_Ventes",
                    "dbo.Dim_Produit",
                ],
                "columns_used": [
                    "dbo.Fact_Ventes.Marge",
                    "dbo.Fact_Ventes.Produit_ID",
                    "dbo.Dim_Produit.Produit_ID",
                    "dbo.Dim_Produit.Nom_Produit",
                ],
                "assumptions": [],
                "clarification_question": "",
            }

        # ---------------------------------------------------------
        # Quantité par produit
        # ---------------------------------------------------------

        if (
            ("quantité" in user_message or "quantite" in user_message)
            and "produit" in user_message
        ):
            return {
                "status": "ok",
                "sql": """
SELECT
    dp.Nom_Produit,
    SUM(fv.Quantite) AS Quantite
FROM dbo.Fact_Ventes AS fv
INNER JOIN dbo.Dim_Produit AS dp
    ON fv.Produit_ID = dp.Produit_ID
GROUP BY
    dp.Nom_Produit
ORDER BY
    Quantite DESC;
""".strip(),
                "explanation": (
                    "Calcul de la quantité vendue totale par produit."
                ),
                "tables_used": [
                    "dbo.Fact_Ventes",
                    "dbo.Dim_Produit",
                ],
                "columns_used": [
                    "dbo.Fact_Ventes.Quantite",
                    "dbo.Fact_Ventes.Produit_ID",
                    "dbo.Dim_Produit.Produit_ID",
                    "dbo.Dim_Produit.Nom_Produit",
                ],
                "assumptions": [
                    "La quantité vendue correspond à la somme de "
                    "Fact_Ventes.Quantite."
                ],
                "clarification_question": "",
            }

        # ---------------------------------------------------------
        # Cas par défaut
        # ---------------------------------------------------------

        return {
            "status": "clarification_needed",
            "sql": "",
            "explanation": (
                "La question ne correspond pas "
                "à un scénario de test connu."
            ),
            "tables_used": [],
            "columns_used": [],
            "assumptions": [],
            "clarification_question": (
                "Pouvez-vous préciser l'indicateur "
                "et la dimension à analyser ?"
            ),
        }
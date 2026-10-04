class FakeLLMClient:
    def generate_structured_response(
        self,
        messages,
        json_schema,
    ):
        return {
            "status": "ok",
            "sql": """
SELECT
    dr.Region,
    SUM(fv.Chiffre_Affaires) AS Chiffre_Affaires_Total
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
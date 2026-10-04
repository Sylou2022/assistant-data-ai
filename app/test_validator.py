from app.sql.validator import SQLValidator


def main() -> None:
    allowed_tables = {
        "dbo.fact_ventes",
        "dbo.dim_region",
    }

    allowed_columns = {
        "dbo.fact_ventes.chiffre_affaires",
        "dbo.fact_ventes.region_id",
        "dbo.dim_region.region_id",
        "dbo.dim_region.region",
        "fact_ventes.chiffre_affaires",
        "fact_ventes.region_id",
        "dim_region.region_id",
        "dim_region.region",
        "chiffre_affaires",
        "region_id",
        "region",
    }

    validator = SQLValidator(
        allowed_tables=allowed_tables,
        allowed_columns=allowed_columns,
    )

    sql = """
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
"""

    result = validator.validate(sql)

    print("Requête valide :", result.is_valid)
    print("Erreurs :", result.errors)
    print("Avertissements :", result.warnings)
    print("SQL normalisé :")
    print(result.normalized_sql)


if __name__ == "__main__":
    main()
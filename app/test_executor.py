from app.database.connection import get_engine
from app.sql.executor import SQLExecutor


def main() -> None:
    executor = SQLExecutor(
        engine=get_engine(),
        max_rows=10,
        dry_run=False,
    )

    sql = """
SELECT TOP (10)
    fv.Vente_ID,
    fv.Chiffre_Affaires
FROM dbo.Fact_Ventes AS fv
ORDER BY
    fv.Vente_ID;
"""

    rows = executor.execute(sql)

    print(f"Lignes retournées : {len(rows)}")

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
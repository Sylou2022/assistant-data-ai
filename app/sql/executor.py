from __future__ import annotations


from typing import Any


import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


class SQLExecutor:
    def __init__(
        self,
        engine: Engine,
        max_rows: int = 1000,
        dry_run: bool = False,
    ) -> None:
        if max_rows <= 0:
            raise ValueError(
                "max_rows doit être supérieur à zéro."
            )

        self.engine = engine
        self.max_rows = max_rows
        self.dry_run = dry_run

    def execute(
        self,
        sql: str,
        parameters: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Exécute la requête et retourne un DataFrame.
        """
        if not sql or not sql.strip():
            raise ValueError(
                "La requête SQL ne peut pas être vide."
            )

        self._ensure_read_only(sql)

        if self.dry_run:
            print("DRY RUN : requête non exécutée.")
            print(sql)
            return pd.DataFrame()

        parameters = parameters or {}

        with self.engine.connect() as connection:
            return pd.read_sql(
                text(sql),
                connection,
                params=parameters,
            )

    def execute_dataframe(
        self,
        sql: str,
        parameters: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Alias de execute() pour compatibilité.
        """
        return self.execute(
            sql,
            parameters=parameters,
        )

    @staticmethod
    def _ensure_read_only(sql: str) -> None:
        """
        Vérification supplémentaire avant l'accès à SQL Server.

        Le SQLValidator doit déjà avoir contrôlé la requête.
        Cette méthode constitue une deuxième protection.
        """
        stripped_sql = sql.strip()

        first_word = stripped_sql.split(
            maxsplit=1
        )[0].upper()

        if first_word not in {"SELECT", "WITH"}:
            raise PermissionError(
                "Seules les requêtes SELECT ou WITH "
                "sont autorisées."
            )

        dangerous_words = {
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "ALTER",
            "TRUNCATE",
            "CREATE",
            "MERGE",
            "EXEC",
            "EXECUTE",
            "GRANT",
            "REVOKE",
        }

        sql_words = set(
            stripped_sql.upper()
            .replace("(", " ")
            .replace(")", " ")
            .replace(";", " ")
            .replace(",", " ")
            .split()
        )

        dangerous_found = sql_words.intersection(
            dangerous_words
        )

        if dangerous_found:
            raise PermissionError(
                "Instruction dangereuse détectée : "
                + ", ".join(sorted(dangerous_found))
            )
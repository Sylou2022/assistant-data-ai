from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import expressions as exp


DANGEROUS_KEYWORDS = {
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
    "DENY",
}


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    normalized_sql: str = ""


class SQLValidator:
    def __init__(
        self,
        allowed_tables: set[str],
        allowed_columns: set[str],
        allowed_relationships: set[tuple[str, str]] | None = None,
    ) -> None:
        self.allowed_tables = {
            self._normalize_identifier(table)
            for table in allowed_tables
        }

        self.allowed_columns = {
            self._normalize_identifier(column)
            for column in allowed_columns
        }

        self.allowed_relationships = {
            (
                self._normalize_identifier(left),
                self._normalize_identifier(right),
            )
            for left, right in (
                allowed_relationships or set()
            )
        }

    def validate(self, sql: str) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        if not sql or not sql.strip():
            return ValidationResult(
                is_valid=False,
                errors=["La requête SQL est vide."],
                warnings=[],
            )

        clean_sql = sql.strip()

        self._check_comments(clean_sql, errors)
        self._check_dangerous_keywords(clean_sql, errors)
        self._check_statement_start(clean_sql, errors)

        parsed = self._parse_sql(
            clean_sql,
            errors,
        )

        if parsed is None:
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
            )

        self._check_single_statement(
            clean_sql,
            errors,
        )

        self._check_select_star(
            parsed,
            errors,
        )

        referenced_tables = self._extract_tables(parsed)
        referenced_columns = self._extract_columns(parsed)
        select_aliases = self._extract_select_aliases(parsed)

        self._check_tables(
            referenced_tables,
            errors,
        )

        self._check_columns(
            referenced_columns,
            select_aliases,
            errors,
        )

        self._check_limit(
            parsed,
            warnings,
        )

        normalized_sql = sqlglot.transpile(
            clean_sql,
            read="tsql",
            write="tsql",
            pretty=True,
        )[0]

        return ValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings,
            normalized_sql=normalized_sql,
        )

    @staticmethod
    def _parse_sql(
        sql: str,
        errors: list[str],
    ) -> exp.Expression | None:
        try:
            statements = sqlglot.parse(
                sql,
                read="tsql",
            )
        except Exception as error:
            errors.append(
                f"Syntaxe SQL invalide : {error}"
            )
            return None

        if not statements:
            errors.append(
                "Aucune instruction SQL détectée."
            )
            return None

        return statements[0]

    @staticmethod
    def _check_comments(
        sql: str,
        errors: list[str],
    ) -> None:
        if "--" in sql or "/*" in sql or "*/" in sql:
            errors.append(
                "Les commentaires SQL ne sont pas autorisés."
            )

    def _check_dangerous_keywords(
        self,
        sql: str,
        errors: list[str],
    ) -> None:
        words = set(
            re.findall(
                r"\b[A-Z]+\b",
                sql.upper(),
            )
        )

        found = words.intersection(
            DANGEROUS_KEYWORDS
        )

        if found:
            keywords = ", ".join(sorted(found))
            errors.append(
                f"Mots-clés dangereux détectés : {keywords}."
            )

    @staticmethod
    def _check_statement_start(
        sql: str,
        errors: list[str],
    ) -> None:
        first_word_match = re.match(
            r"^\s*([A-Za-z]+)",
            sql,
        )

        if not first_word_match:
            errors.append(
                "Impossible d'identifier le début de la requête."
            )
            return

        first_word = first_word_match.group(1).upper()

        if first_word not in {"SELECT", "WITH"}:
            errors.append(
                "La requête doit commencer par SELECT ou WITH."
            )

    @staticmethod
    def _check_single_statement(
        sql: str,
        errors: list[str],
    ) -> None:
        without_final_semicolon = sql.rstrip().rstrip(";")

        if ";" in without_final_semicolon:
            errors.append(
                "Une seule instruction SQL est autorisée."
            )

    @staticmethod
    def _check_select_star(
        expression: exp.Expression,
        errors: list[str],
    ) -> None:
        stars = list(expression.find_all(exp.Star))

        if stars:
            errors.append(
                "SELECT * n'est pas autorisé."
            )

    def _check_tables(
        self,
        tables: set[str],
        errors: list[str],
    ) -> None:
        for table in tables:
            if table not in self.allowed_tables:
                errors.append(
                    f"Table non autorisée : {table}."
                )

    def _check_columns(
        self,
        columns: set[str],
        select_aliases: set[str],
        errors: list[str],
    ) -> None:
        matching_columns = {
            allowed.split(".")[-1]
            for allowed in self.allowed_columns
        }

        for column in columns:
            if column in {"", "*"}:
                continue

            if column in select_aliases:
                continue

            if column in self.allowed_columns:
                continue

            column_name = column.split(".")[-1]

            if column_name in select_aliases:
                continue

            if column_name in matching_columns:
                continue

            errors.append(
                "Colonne non autorisée ou inconnue : "
                f"{column}."
            )

    @staticmethod
    def _check_limit(
        expression: exp.Expression,
        warnings: list[str],
    ) -> None:
        if not expression.find(exp.Limit):
            warnings.append(
                "La requête ne contient pas de limite de lignes."
            )

    @staticmethod
    def _extract_tables(
        expression: exp.Expression,
    ) -> set[str]:
        tables: set[str] = set()

        for table in expression.find_all(exp.Table):
            parts = [
                part
                for part in [
                    table.catalog,
                    table.db,
                    table.name,
                ]
                if part
            ]

            tables.add(
                ".".join(
                    str(part).lower()
                    for part in parts
                )
            )

        return tables

    @staticmethod
    def _extract_columns(
        expression: exp.Expression,
    ) -> set[str]:
        columns: set[str] = set()

        for column in expression.find_all(exp.Column):
            table_name = column.table
            column_name = column.name

            if table_name:
                columns.add(
                    f"{table_name}.{column_name}".lower()
                )
            else:
                columns.add(
                    column_name.lower()
                )

        return columns

    @staticmethod
    def _normalize_identifier(
        identifier: Any,
    ) -> str:
        return (
            str(identifier)
            .replace("[", "")
            .replace("]", "")
            .strip()
            .lower()
        )

    @staticmethod
    def _extract_select_aliases(
        expression: exp.Expression,
    ) -> set[str]:
        aliases: set[str] = set()

        for alias in expression.find_all(exp.Alias):
            alias_name = alias.alias

            if alias_name:
                aliases.add(
                    str(alias_name).lower()
                )

        return aliases
from __future__ import annotations

from typing import Any


def _normalize_identifier(
    value: str,
) -> str:
    return (
        str(value)
        .replace("[", "")
        .replace("]", "")
        .strip()
        .lower()
    )


def build_allowed_tables(
    catalog: dict[str, Any],
) -> set[str]:
    allowed_tables: set[str] = set()

    for table in catalog.get("tables", []):
        schema = _normalize_identifier(
            table.get("schema", "dbo")
        )

        name = _normalize_identifier(
            table.get("name", "")
        )

        if not name:
            continue

        allowed_tables.add(name)
        allowed_tables.add(
            f"{schema}.{name}"
        )

    for item in catalog.get(
        "ai_metadata",
        {}
    ).get(
        "allowed_tables",
        [],
    ):
        table_name = item.get(
            "Table_Name",
            ""
        )

        if table_name:
            allowed_tables.add(
                _normalize_identifier(table_name)
            )

    return allowed_tables


def build_allowed_columns(
    catalog: dict[str, Any],
) -> set[str]:
    allowed_columns: set[str] = set()

    for table in catalog.get("tables", []):
        schema = _normalize_identifier(
            table.get("schema", "dbo")
        )

        table_name = _normalize_identifier(
            table.get("name", "")
        )

        if not table_name:
            continue

        for column in table.get(
            "columns",
            [],
        ):
            column_name = _normalize_identifier(
                column.get("name", "")
            )

            if not column_name:
                continue

            allowed_columns.add(
                column_name
            )

            allowed_columns.add(
                f"{table_name}.{column_name}"
            )

            allowed_columns.add(
                f"{schema}.{table_name}.{column_name}"
            )

    return allowed_columns


def build_allowed_relationships(
    catalog: dict[str, Any],
) -> set[tuple[str, str]]:
    relationships: set[tuple[str, str]] = set()

    for relation in catalog.get(
        "relationships",
        [],
    ):
        source_table = _normalize_identifier(
            relation.get("source_table", "")
        )

        source_column = _normalize_identifier(
            relation.get("source_column", "")
        )

        target_table = _normalize_identifier(
            relation.get("target_table", "")
        )

        target_column = _normalize_identifier(
            relation.get("target_column", "")
        )

        if not source_table or not target_table:
            continue

        relationships.add(
            (
                f"{source_table}.{source_column}",
                f"{target_table}.{target_column}",
            )
        )

    ai_relationships = (
        catalog
        .get("ai_metadata", {})
        .get("relationships", [])
    )

    for relation in ai_relationships:
        source_table = _normalize_identifier(
            relation.get("Source_Table", "")
        )

        source_column = _normalize_identifier(
            relation.get("Source_Column", "")
        )

        target_table = _normalize_identifier(
            relation.get("Target_Table", "")
        )

        target_column = _normalize_identifier(
            relation.get("Target_Column", "")
        )

        if not source_table or not target_table:
            continue

        relationships.add(
            (
                f"{source_table}.{source_column}",
                f"{target_table}.{target_column}",
            )
        )

    return relationships
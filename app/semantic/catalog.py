from datetime import datetime, timezone
from typing import Any

from sqlalchemy.engine import Engine

from app.database.queries import (
    get_columns,
    get_foreign_keys,
    get_primary_keys,
    get_tables,
)


def build_catalog(engine: Engine) -> dict[str, Any]:
    tables = get_tables(engine)
    columns = get_columns(engine)
    primary_keys = get_primary_keys(engine)
    foreign_keys = get_foreign_keys(engine)

    primary_key_set = {
        (
            key["TABLE_SCHEMA"],
            key["TABLE_NAME"],
            key["COLUMN_NAME"],
        )
        for key in primary_keys
    }

    catalog: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tables": [],
        "relationships": foreign_keys,
    }

    for table in tables:
        schema_name = table["TABLE_SCHEMA"]
        table_name = table["TABLE_NAME"]

        table_columns = [
            {
                "name": column["COLUMN_NAME"],
                "data_type": column["DATA_TYPE"],
                "nullable": column["IS_NULLABLE"] == "YES",
                "position": column["ORDINAL_POSITION"],
                "is_primary_key": (
                    schema_name,
                    table_name,
                    column["COLUMN_NAME"],
                ) in primary_key_set,
            }
            for column in columns
            if (
                column["TABLE_SCHEMA"] == schema_name
                and column["TABLE_NAME"] == table_name
            )
        ]

        catalog["tables"].append(
            {
                "schema": schema_name,
                "name": table_name,
                "columns": table_columns,
            }
        )

    return catalog
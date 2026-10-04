from sqlalchemy import text
from sqlalchemy.engine import Engine


def get_tables(engine: Engine) -> list[dict]:
    query = text(
        """
        SELECT
            TABLE_SCHEMA,
            TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_SCHEMA, TABLE_NAME
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [dict(row) for row in rows]


def get_columns(engine: Engine) -> list[dict]:
    query = text(
        """
        SELECT
            TABLE_SCHEMA,
            TABLE_NAME,
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            ORDINAL_POSITION
        FROM INFORMATION_SCHEMA.COLUMNS
        ORDER BY
            TABLE_SCHEMA,
            TABLE_NAME,
            ORDINAL_POSITION
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [dict(row) for row in rows]


def get_primary_keys(engine: Engine) -> list[dict]:
    query = text(
        """
        SELECT
            TABLE_SCHEMA,
            TABLE_NAME,
            COLUMN_NAME,
            ORDINAL_POSITION
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE OBJECTPROPERTY(
            OBJECT_ID(
                QUOTENAME(TABLE_SCHEMA)
                + '.'
                + QUOTENAME(TABLE_NAME)
            ),
            'IsUserTable'
        ) = 1
        AND CONSTRAINT_NAME LIKE 'PK%'
        ORDER BY
            TABLE_SCHEMA,
            TABLE_NAME,
            ORDINAL_POSITION
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [dict(row) for row in rows]


def get_foreign_keys(engine: Engine) -> list[dict]:
    query = text(
        """
        SELECT
            OBJECT_SCHEMA_NAME(fk.parent_object_id)
                AS source_schema,
            OBJECT_NAME(fk.parent_object_id)
                AS source_table,
            COL_NAME(
                fkc.parent_object_id,
                fkc.parent_column_id
            ) AS source_column,
            OBJECT_SCHEMA_NAME(fk.referenced_object_id)
                AS target_schema,
            OBJECT_NAME(fk.referenced_object_id)
                AS target_table,
            COL_NAME(
                fkc.referenced_object_id,
                fkc.referenced_column_id
            ) AS target_column
        FROM sys.foreign_keys AS fk
        INNER JOIN sys.foreign_key_columns AS fkc
            ON fk.object_id = fkc.constraint_object_id
        ORDER BY
            source_schema,
            source_table,
            source_column
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [dict(row) for row in rows]

def get_allowed_tables(engine: Engine) -> list[dict]:
    query = text(
        """
        SELECT
            Table_ID,
            Table_Name,
            Table_Type,
            Usage_Description,
            Is_Enabled
        FROM dbo.AI_Allowed_Tables
        WHERE Is_Enabled = 1
        ORDER BY Table_Name
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [dict(row) for row in rows]


def get_business_dictionary(engine: Engine) -> list[dict]:
    query = text(
        """
        SELECT
            Dictionnaire_ID,
            Table_Name,
            Column_Name,
            Nom_Metier,
            Description_Metier,
            Type_Metier,
            Unite,
            Regle_Metier,
            Exemple,
            Sensible,
            Utilisable_IA
        FROM dbo.Data_Dictionnaire_Metier
        WHERE Utilisable_IA = 1
        AND ISNULL(Sensible, 0) = 0
        ORDER BY Table_Name, Column_Name
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [dict(row) for row in rows]


def get_kpis(engine: Engine) -> list[dict]:
    query = text(
        """
        SELECT
            KPI_ID,
            Nom_KPI,
            Description,
            Formule_SQL,
            Unite,
            Frequence
        FROM dbo.Business_KPI
        ORDER BY Nom_KPI
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [dict(row) for row in rows]


def get_ai_rules(engine: Engine) -> list[dict]:
    query = text(
        """
        SELECT
            Rule_ID,
            Rule_Name,
            Rule_Type,
            Rule_Value,
            Description
        FROM dbo.AI_SQL_Rules
        WHERE Is_Active = 1
        ORDER BY Rule_ID
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [dict(row) for row in rows]


def get_sql_examples(engine: Engine) -> list[dict]:
    query = text(
        """
        SELECT
            Example_ID,
            Question,
            Intent,
            SQL_Query,
            Explanation
        FROM dbo.AI_SQL_Examples
        WHERE Is_Active = 1
        ORDER BY Example_ID
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [dict(row) for row in rows]


def get_semantic_relationships(
    engine: Engine,
) -> list[dict]:
    query = text(
        """
        SELECT
            Relationship_ID,
            Source_Table,
            Source_Column,
            Target_Table,
            Target_Column,
            Relationship_Type,
            Description
        FROM dbo.Semantic_Relationships
        WHERE Is_AI_Enabled = 1
        ORDER BY Relationship_ID
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [dict(row) for row in rows]


def get_semantic_synonyms(
    engine: Engine,
) -> list[dict]:
    query = text(
        """
        SELECT
            s.Synonym_ID,
            s.Entity_ID,
            s.Synonym
        FROM dbo.Semantic_Synonyms AS s
        INNER JOIN dbo.Semantic_Entities AS e
            ON e.Entity_ID = s.Entity_ID
        WHERE e.Is_AI_Enabled = 1
        ORDER BY s.Entity_ID, s.Synonym
        """
    )

    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()

    return [dict(row) for row in rows]
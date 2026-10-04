from app.llm.prompts import (
    SQL_SYSTEM_PROMPT,
    build_sql_json_schema,
    build_sql_messages,
    build_sql_user_prompt,
)


def test_system_prompt_contains_sql_rules():
    assert "SELECT" in SQL_SYSTEM_PROMPT
    assert "INSERT" in SQL_SYSTEM_PROMPT
    assert "SELECT *" in SQL_SYSTEM_PROMPT


def test_user_prompt_contains_question_and_context():
    prompt = build_sql_user_prompt(
        question="Quel est le chiffre d'affaires ?",
        semantic_context="Fact_Ventes.Chiffre_Affaires",
    )

    assert "Quel est le chiffre d'affaires ?" in prompt
    assert "Fact_Ventes.Chiffre_Affaires" in prompt


def test_messages_have_system_and_user_roles():
    messages = build_sql_messages(
        question="Quel est le chiffre d'affaires ?",
        semantic_context="Fact_Ventes.Chiffre_Affaires",
    )

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_json_schema_contains_required_fields():
    schema = build_sql_json_schema()
    properties = schema["schema"]["properties"]

    assert "status" in properties
    assert "sql" in properties
    assert "tables_used" in properties
    assert "clarification_question" in properties
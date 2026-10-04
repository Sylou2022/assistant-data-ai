from app.llm.fake_client import FakeLLMClient


def build_messages(question: str):
    return [
        {
            "role": "system",
            "content": "Instructions de test",
        },
        {
            "role": "user",
            "content": question,
        },
    ]


def test_revenue_by_region():
    client = FakeLLMClient()

    result = client.generate_structured_response(
        messages=build_messages(
            "Quel est le chiffre d'affaires par région ?"
        ),
        json_schema={},
    )

    assert result["status"] == "ok"
    assert "Fact_Ventes" in result["sql"]
    assert "Dim_Region" in result["sql"]
    assert "Chiffre_Affaires" in result["sql"]


def test_margin_by_product():
    client = FakeLLMClient()

    result = client.generate_structured_response(
        messages=build_messages(
            "Quelle est la marge totale par produit ?"
        ),
        json_schema={},
    )

    assert result["status"] == "ok"
    assert "Fact_Ventes" in result["sql"]
    assert "Dim_Produit" in result["sql"]
    assert "Marge" in result["sql"]


def test_quantity_by_product():
    client = FakeLLMClient()

    result = client.generate_structured_response(
        messages=build_messages(
            "Quelle est la quantité vendue par produit ?"
        ),
        json_schema={},
    )

    assert result["status"] == "ok"
    assert "Fact_Ventes" in result["sql"]
    assert "Dim_Produit" in result["sql"]
    assert "Quantite" in result["sql"]


def test_unknown_question_returns_clarification():
    client = FakeLLMClient()

    result = client.generate_structured_response(
        messages=build_messages(
            "Quel est le taux de satisfaction client ?"
        ),
        json_schema={},
    )

    assert result["status"] == "clarification_needed"
    assert result["sql"] == ""
    assert result["clarification_question"]
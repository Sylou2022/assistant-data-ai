from app.llm.fake_client import FakeLLMClient
from app.sql.generator import SQLGenerator


def test_generator_uses_fake_client():
    generator = SQLGenerator(
        llm_client=FakeLLMClient(),
    )

    result = generator.generate(
        question="Quelle est la marge totale par produit ?",
        semantic_context=(
            "Fact_Ventes, Dim_Produit, Marge, "
            "Nom_Produit"
        ),
    )

    assert result["status"] == "ok"
    assert "SUM(fv.Marge)" in result["sql"]
    assert "dbo.Dim_Produit" in result["sql"]
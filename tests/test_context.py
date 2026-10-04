from pathlib import Path


def test_context_contains_business_terms():
    context_path = Path("semantic_context.txt")
    context = context_path.read_text(encoding="utf-8")

    assert "Chiffre_Affaires" in context
    assert "Fact_Ventes" in context
    assert "Dim_Region" in context
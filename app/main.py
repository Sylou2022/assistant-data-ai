import json
from pathlib import Path
from typing import Any
from app.llm.prompts import (
    build_prompt_preview,
)

from app.database.connection import get_engine
from app.database.queries import (
    get_ai_rules,
    get_allowed_tables,
    get_business_dictionary,
    get_kpis,
    get_semantic_relationships,
    get_semantic_synonyms,
    get_sql_examples,
)
from app.semantic.catalog import build_catalog
from app.semantic.context import SemanticContextBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CATALOG_PATH = PROJECT_ROOT / "semantic_catalog.json"
CONTEXT_PATH = PROJECT_ROOT / "semantic_context.txt"


def load_semantic_layer() -> dict[str, Any]:
    engine = get_engine()

    catalog = build_catalog(engine)

    allowed_tables = get_allowed_tables(engine)
    business_dictionary = get_business_dictionary(engine)
    kpis = get_kpis(engine)
    rules = get_ai_rules(engine)
    examples = get_sql_examples(engine)
    relationships = get_semantic_relationships(engine)
    synonyms = get_semantic_synonyms(engine)

    catalog["ai_metadata"] = {
        "allowed_tables": allowed_tables,
        "business_dictionary": business_dictionary,
        "kpis": kpis,
        "rules": rules,
        "examples": examples,
        "relationships": relationships,
        "synonyms": synonyms,
    }

    CATALOG_PATH.write_text(
        json.dumps(
            catalog,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    return catalog


def build_llm_context(
    catalog: dict[str, Any],
) -> str:
    metadata = catalog.get("ai_metadata", {})

    context_builder = SemanticContextBuilder(
        catalog=catalog,
        allowed_tables=metadata.get(
            "allowed_tables",
            [],
        ),
        business_dictionary=metadata.get(
            "business_dictionary",
            [],
        ),
        kpis=metadata.get("kpis", []),
        relationships=metadata.get(
            "relationships",
            [],
        ),
        rules=metadata.get("rules", []),
        examples=metadata.get(
            "examples",
            [],
        ),
        synonyms=metadata.get(
            "synonyms",
            [],
        ),
    )

    return context_builder.build()


def main() -> None:
    catalog = load_semantic_layer()
    llm_context = build_llm_context(catalog)

    CONTEXT_PATH.write_text(
        llm_context,
        encoding="utf-8",
    )

    table_count = len(catalog.get("tables", []))

    column_count = sum(
        len(table.get("columns", []))
        for table in catalog.get("tables", [])
    )

    metadata = catalog.get("ai_metadata", {})

    print("Connexion SQL Server réussie.")
    print("Semantic Layer chargée automatiquement.")
    print(f"Tables détectées : {table_count}")
    print(f"Colonnes détectées : {column_count}")
    print(f"Tables autorisées : {len(metadata.get('allowed_tables', []))}")
    print(f"KPI chargés : {len(metadata.get('kpis', []))}")
    print(
        "Définitions métier chargées : "
        f"{len(metadata.get('business_dictionary', []))}"
    )
    print(f"Règles SQL chargées : {len(metadata.get('rules', []))}")
    print(
        "Exemples SQL chargés : "
        f"{len(metadata.get('examples', []))}"
    )
    print(f"Catalogue sauvegardé dans : {CATALOG_PATH}")
    print(f"Contexte LLM sauvegardé dans : {CONTEXT_PATH}")

    print("\nContexte envoyé au LLM :\n")
    print(llm_context)

    
    question = "Quel est le chiffre d'affaires par région ?"

    prompt_preview = build_prompt_preview(
        question=question,
        semantic_context=llm_context,
    )

    prompt_path = PROJECT_ROOT / "prompt_preview.json"

    prompt_path.write_text(
        prompt_preview,
        encoding="utf-8",
    )

    print(f"Prompt de test sauvegardé dans : {prompt_path}")


if __name__ == "__main__":
    main()
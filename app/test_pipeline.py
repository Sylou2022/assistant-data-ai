import json
from pathlib import Path

from app.llm.fake_client import FakeLLMClient
from app.sql.generator import SQLGenerator
from app.sql.permissions import (
    build_allowed_columns,
    build_allowed_relationships,
    build_allowed_tables,
)
from app.sql.validator import SQLValidator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "semantic_catalog.json"
CONTEXT_PATH = PROJECT_ROOT / "semantic_context.txt"


def main() -> None:
    catalog = json.loads(
        CATALOG_PATH.read_text(
            encoding="utf-8",
        )
    )

    semantic_context = CONTEXT_PATH.read_text(
        encoding="utf-8",
    )

    generator = SQLGenerator(
        llm_client=FakeLLMClient(),
    )

    generated_result = generator.generate(
        question="Quel est le chiffre d'affaires par région ?",
        semantic_context=semantic_context,
    )

    if generated_result["status"] != "ok":
        print(
            generated_result[
                "clarification_question"
            ]
        )
        return

    validator = SQLValidator(
        allowed_tables=build_allowed_tables(
            catalog,
        ),
        allowed_columns=build_allowed_columns(
            catalog,
        ),
        allowed_relationships=build_allowed_relationships(
            catalog,
        ),
    )

    validation_result = validator.validate(
        generated_result["sql"],
    )

    print("SQL généré :")
    print(generated_result["sql"])
    print()
    print("Validation réussie :")
    print(validation_result.is_valid)
    print()
    print("Erreurs :")
    print(validation_result.errors)
    print()
    print("Avertissements :")
    print(validation_result.warnings)


if __name__ == "__main__":
    main()
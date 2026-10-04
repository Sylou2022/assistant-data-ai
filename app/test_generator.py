from pathlib import Path

from app.llm.fake_client import FakeLLMClient
from app.sql.generator import SQLGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATH = PROJECT_ROOT / "semantic_context.txt"


def main() -> None:
    semantic_context = CONTEXT_PATH.read_text(
        encoding="utf-8",
    )

    generator = SQLGenerator(
        llm_client=FakeLLMClient(),
    )

    result = generator.generate(
        question="Quel est le chiffre d'affaires par région ?",
        semantic_context=semantic_context,
    )

    print("Résultat de la génération SQL :")
    print(result["sql"])
    print()
    print("Explication :")
    print(result["explanation"])
    print()
    print("Tables utilisées :")
    print(result["tables_used"])


if __name__ == "__main__":
    main()
from pathlib import Path

from app.sql.generator import SQLGenerator
from tests.fake_llm_client import FakeLLMClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATH = PROJECT_ROOT / "semantic_context.txt"


def main() -> None:
    semantic_context = CONTEXT_PATH.read_text(
        encoding="utf-8",
    )

    # generator = SQLGenerator() 

    generator = SQLGenerator(
        llm_client=FakeLLMClient(),
    )

    result = generator.generate(
        question="Quel est le chiffre d'affaires par région ?",
        semantic_context=semantic_context,
    )

    print("Résultat simulé :")
    print(result)


if __name__ == "__main__":
    main()
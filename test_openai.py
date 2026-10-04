
from pathlib import Path

from app.llm.openai_client import OpenAIClient
from app.sql.generator import SQLGenerator


PROJECT_ROOT = Path(__file__).resolve().parent
CONTEXT_PATH = PROJECT_ROOT / "semantic_context.txt"


def main() -> None:
    semantic_context = CONTEXT_PATH.read_text(
        encoding="utf-8"
    )

    client = OpenAIClient()

    generator = SQLGenerator(
        llm_client=client
    )

    question = (
        "Quel est le chiffre d'affaires "
        "par région ?"
    )

    print()
    print("=" * 60)
    print("QUESTION")
    print("=" * 60)
    print(question)

    print()
    print("Appel à OpenAI en cours...")

    result = generator.generate(
        question=question,
        semantic_context=semantic_context,
    )

    print()
    print("=" * 60)
    print("RÉPONSE DU LLM")
    print("=" * 60)

    print(f"Status : {result['status']}")
    print(f"SQL : {result['sql']}")

    print()
    print(f"Tables utilisées : {result['tables_used']}")
    print(f"Colonnes utilisées : {result['columns_used']}")

    print()
    print(f"Explication : {result['explanation']}")

    print()
    print(f"Hypothèses : {result['assumptions']}")

    if result["clarification_question"]:
        print()
        print(
            "Clarification : "
            f"{result['clarification_question']}"
        )


if __name__ == "__main__":
    main()

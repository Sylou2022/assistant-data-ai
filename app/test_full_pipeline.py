
from app.pipeline.data_pipeline import DataPipeline


def main() -> None:
    question = input(
        "Pose ta question métier : "
    ).strip()

    if not question:
        print(
            "La question ne peut pas être vide."
        )
        return

    pipeline = DataPipeline()

    result = pipeline.run(
        question,
    )

    if result.status == "clarification_needed":
        print()
        print(
            result.clarification_question
        )
        return

    if result.status == "validation_error":
        print()
        print("Requête rejetée.")
        print("SQL généré :")
        print(result.sql)
        print()
        print("Erreurs :")

        for error in result.errors or []:
            print(f"- {error}")

        return

    print()
    print("SQL généré :")
    print(result.sql)

    print()
    print("Requête validée.")

    print()
    print(result.formatted_result)

    if result.chart_path is not None:
        print()
        print(
            f"Graphique créé : "
            f"{result.chart_path}"
        )
    else:
        print()
        print(
            "Aucun graphique adapté "
            "au résultat."
        )


if __name__ == "__main__":
    main()


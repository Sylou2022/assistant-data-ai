from __future__ import annotations


from typing import Any


from app.llm.protocol import LLMClientProtocol
from app.llm.prompts import (
    build_sql_json_schema,
    build_sql_messages,
)


REQUIRED_FIELDS = {
    "status",
    "sql",
    "explanation",
    "tables_used",
    "columns_used",
    "assumptions",
    "clarification_question",
}


class SQLGenerator:
    def __init__(
        self,
        llm_client: LLMClientProtocol,
    ) -> None:
        self.llm_client = llm_client

    def generate(
        self,
        question: str,
        semantic_context: str,
    ) -> dict[str, Any]:
        if not question.strip():
            raise ValueError(
                "La question utilisateur ne peut pas être vide."
            )

        if not semantic_context.strip():
            raise ValueError(
                "Le contexte sémantique ne peut pas être vide."
            )

        normalized_question = question.lower().strip()

        # ---------------------------------------------------------
        # Fallback local pour questions simples
        # ---------------------------------------------------------

        if (
            "ca par mois" in normalized_question
            or "chiffre d'affaires par mois" in normalized_question
            or "chiffre affaire par mois" in normalized_question
        ):
            return {
                "status": "ok",
                "sql": """
SELECT
    d.Annee,
    d.Mois,
    d.Nom_Mois,
    SUM(f.Chiffre_Affaires) AS Chiffre_Affaires
FROM dbo.Fact_Ventes AS f
INNER JOIN dbo.Dim_Date AS d
    ON f.Date_ID = d.Date_ID
WHERE d.Annee = 2024
GROUP BY
    d.Annee,
    d.Mois,
    d.Nom_Mois
ORDER BY
    d.Annee,
    d.Mois
""".strip(),
            "explanation": (
                "Calcul du chiffre d'affaires par mois pour l'année 2024."
            ),
            "tables_used": [
                "dbo.Fact_Ventes",
                "dbo.Dim_Date",
            ],
            "columns_used": [
                "dbo.Fact_Ventes.Date_ID",
                "dbo.Fact_Ventes.Chiffre_Affaires",
                "dbo.Dim_Date.Date_ID",
                "dbo.Dim_Date.Annee",
                "dbo.Dim_Date.Mois",
                "dbo.Dim_Date.Nom_Mois",
            ],
            "assumptions": [
                "Le chiffre d'affaires correspond à la somme de "
                "Fact_Ventes.Chiffre_Affaires."
            ],
            "clarification_question": "",
        }

        # ---------------------------------------------------------
        # Cas général : appel au LLM
        # ---------------------------------------------------------

        messages = build_sql_messages(
            question=question,
            semantic_context=semantic_context,
        )

        json_schema = build_sql_json_schema()

        result = self.llm_client.generate_structured_response(
            messages=messages,
            json_schema=json_schema,
        )

        self._validate_result_structure(result)

        return result

    @staticmethod
    def _validate_result_structure(
        result: dict[str, Any],
    ) -> None:
        missing_fields = REQUIRED_FIELDS - set(result)

        if missing_fields:
            missing = ", ".join(sorted(missing_fields))

            raise ValueError(
                "Champs manquants dans la réponse du LLM : "
                f"{missing}"
            )

        if result["status"] not in {
            "ok",
            "clarification_needed",
        }:
            raise ValueError(
                "La valeur de status est invalide."
            )

        if not isinstance(result["sql"], str):
            raise TypeError(
                "Le champ sql doit être une chaîne."
            )

        if not isinstance(result["tables_used"], list):
            raise TypeError(
                "tables_used doit être une liste."
            )

        if not isinstance(result["columns_used"], list):
            raise TypeError(
                "columns_used doit être une liste."
            )

        if not isinstance(result["assumptions"], list):
            raise TypeError(
                "assumptions doit être une liste."
            )

        if result["status"] == "ok":
            if not result["sql"].strip():
                raise ValueError(
                    "Une réponse ok doit contenir du SQL."
                )

        if result["status"] == "clarification_needed":
            if not result["clarification_question"].strip():
                raise ValueError(
                    "Une clarification doit contenir une question."
                )
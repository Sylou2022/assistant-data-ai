from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "La variable OPENAI_API_KEY est absente "
                "du fichier .env."
            )

        self.model = model or os.getenv(
            "OPENAI_MODEL",
            "gpt-5.5",
        )

        self.client = OpenAI(
            api_key=api_key,
        )

    def generate_structured_response(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        response = self.client.responses.create(
            model=self.model,
            input=messages,
            text={
                "format": {
                    "type": "json_schema",
                    "name": json_schema["name"],
                    "strict": json_schema["strict"],
                    "schema": json_schema["schema"],
                }
            },
        )

        output_text = response.output_text

        if not output_text:
            raise RuntimeError(
                "Le modèle LLM n'a retourné aucun contenu."
            )

        try:
            result = json.loads(output_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                "La réponse du LLM n'est pas un JSON valide."
            ) from error

        return result
# code : llm/openai_client.py

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Client OpenAI pour les réponses structurées."""

    def __init__(
        self,
        model: str = "gpt-5.6-luna",
        reasoning_effort: str | None = "low",
        timeout: float = 45.0,
        max_retries: int = 1,
    ) -> None:
        """
        reasoning_effort :
            Pour du text-to-SQL, "low" (ou "none") suffit presque toujours.
            Le défaut du modèle ("medium") ajoute plusieurs secondes de
            réflexion à chaque question. None = ne pas envoyer le paramètre.

        timeout / max_retries :
            Évitent qu'un appel bloqué ou une erreur 429 fasse attendre
            l'utilisateur pendant des minutes.
        """
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "La variable OPENAI_API_KEY est absente."
            )

        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def generate_structured_response(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Génère une réponse JSON conforme au schéma fourni."""

        request: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "text": {
                "format": {
                    "type": "json_schema",
                    **json_schema,
                },
            },
        }

        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}

        start = time.perf_counter()

        try:
            response = self.client.responses.create(**request)

        except BadRequestError as error:
            # Le modèle refuse le paramètre « reasoning » : on réessaie sans.
            if "reasoning" not in request or "reasoning" not in str(error).lower():
                raise

            logger.warning(
                "Paramètre reasoning refusé par %s : nouvel essai sans.",
                self.model,
            )
            self.reasoning_effort = None
            request.pop("reasoning")
            response = self.client.responses.create(**request)

        self._log_usage(response, time.perf_counter() - start)

        try:
            return json.loads(response.output_text)

        except json.JSONDecodeError as error:
            raise ValueError(
                "OpenAI a retourné une réponse "
                "qui n'est pas un JSON valide."
            ) from error

    def _log_usage(self, response: Any, elapsed: float) -> None:
        """Journalise durée et tokens : montre si le temps part dans
        le prompt (input) ou dans la réflexion (reasoning)."""

        usage = getattr(response, "usage", None)

        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        reasoning_tokens = getattr(
            getattr(usage, "output_tokens_details", None),
            "reasoning_tokens",
            None,
        )

        print(
            f"[PERF] LLM {self.model} "
            f"(effort={self.reasoning_effort}): {elapsed:.2f}s | "
            f"tokens entrée={input_tokens} sortie={output_tokens} "
            f"dont raisonnement={reasoning_tokens}"
        )
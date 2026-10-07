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
    """Client OpenAI : réponses structurées (JSON) et conversation avec outils."""

    def __init__(
        self,
        model: str = "gpt-5.6-luna",
        reasoning_effort: str | None = "low",
        timeout: float = 60.0,
        max_retries: int = 1,
    ) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("La variable OPENAI_API_KEY est absente.")

        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)

    def _create(self, request: dict[str, Any]) -> Any:
        """Appel Responses API, avec repli si « reasoning » est refusé."""
        try:
            return self.client.responses.create(**request)
        except BadRequestError as error:
            if "reasoning" not in request or "reasoning" not in str(error).lower():
                raise
            logger.warning("Paramètre reasoning refusé par %s : nouvel essai sans.", self.model)
            self.reasoning_effort = None
            request.pop("reasoning")
            return self.client.responses.create(**request)

    def respond(
        self,
        input_items: list,
        tools: list[dict] | None = None,
        max_output_tokens: int = 4000,
    ) -> Any:
        """Un tour de conversation : texte et/ou appels d'outils."""
        request: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": max_output_tokens,
        }
        if tools:
            request["tools"] = tools
        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}

        start = time.perf_counter()
        response = self._create(request)
        self._log_usage(response, time.perf_counter() - start)
        return response

    def generate_structured_response(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Réponse JSON conforme au schéma (utilisé par l'ancien pipeline SQL)."""
        request: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "text": {"format": {"type": "json_schema", **json_schema}},
        }
        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}

        start = time.perf_counter()
        response = self._create(request)
        self._log_usage(response, time.perf_counter() - start)

        try:
            return json.loads(response.output_text)
        except json.JSONDecodeError as error:
            raise ValueError("OpenAI a retourné une réponse qui n'est pas un JSON valide.") from error

    def _log_usage(self, response: Any, elapsed: float) -> None:
        usage = getattr(response, "usage", None)
        reasoning_tokens = getattr(
            getattr(usage, "output_tokens_details", None), "reasoning_tokens", None
        )
        print(
            f"[PERF] LLM {self.model} (effort={self.reasoning_effort}): {elapsed:.2f}s | "
            f"tokens entrée={getattr(usage, 'input_tokens', None)} "
            f"sortie={getattr(usage, 'output_tokens', None)} "
            f"dont raisonnement={reasoning_tokens}"
        )
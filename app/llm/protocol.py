
from __future__ import annotations

from typing import Any, Protocol


class LLMClientProtocol(Protocol):
    """Contrat commun pour les clients LLM."""

    def generate_structured_response(
        self,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        ...


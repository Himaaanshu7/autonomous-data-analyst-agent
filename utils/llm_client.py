import re
import json
import time
import logging
from typing import Any

import anthropic

from config.settings import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around the Anthropic Messages API with retry + prompt caching."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.llm_model
        self._max_tokens = settings.llm_max_tokens

    # ------------------------------------------------------------------
    # Core completion
    # ------------------------------------------------------------------

    def complete(
        self,
        system: str,
        user: str,
        cache_system: bool = False,
        max_retries: int = 3,
    ) -> str:
        """Return raw text from the model.

        When cache_system=True the system prompt is sent with an ephemeral
        cache_control block, saving tokens on repeated calls with the same
        schema context.
        """
        system_content: Any = system
        if cache_system:
            system_content = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system_content,
                    messages=[{"role": "user", "content": user}],
                )
                return response.content[0].text
            except anthropic.RateLimitError as exc:
                wait = 2 ** attempt
                logger.warning("Rate limited — retrying in %ds (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
                last_error = exc
            except anthropic.APIStatusError as exc:
                logger.error("API error on attempt %d: %s", attempt + 1, exc)
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(1)

        raise RuntimeError(f"LLM request failed after {max_retries} retries: {last_error}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def complete_json(
        self,
        system: str,
        user: str,
        cache_system: bool = False,
    ) -> dict:
        """Return parsed JSON dict from the model response."""
        raw = self.complete(system, user, cache_system)
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        text = match.group(1).strip() if match else raw.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Attempt to find a JSON object anywhere in the text
            obj_match = re.search(r"\{[\s\S]*\}", text)
            if obj_match:
                return json.loads(obj_match.group(0))
            logger.error("Non-JSON response: %s", text[:300])
            raise ValueError(f"Model returned non-JSON output: {text[:300]}")


llm_client = LLMClient()

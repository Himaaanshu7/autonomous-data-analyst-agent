import re
import json
import time
import logging
from typing import Any

from groq import Groq, RateLimitError, APIStatusError

from config.settings import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Groq API wrapper — free tier, no billing required.

    Default model: llama-3.3-70b-versatile
    Free limits: ~14,400 req/day, 6,000 tokens/min
    """

    def __init__(self) -> None:
        self._client: Groq | None = None
        self._model = settings.llm_model
        self._max_tokens = settings.llm_max_tokens

    def _get_client(self) -> Groq:
        """Lazy-initialize Groq client so st.secrets are loaded first."""
        if self._client is None:
            import os
            # Try st.secrets directly at call time (cloud deploy)
            try:
                import streamlit as st
                key = st.secrets.get("GROQ_API_KEY", "")
                if key:
                    os.environ["GROQ_API_KEY"] = key
            except Exception:
                pass
            api_key = os.environ.get("GROQ_API_KEY", settings.groq_api_key)
            self._client = Groq(api_key=api_key)
        return self._client

    def complete(
        self,
        system: str,
        user: str,
        cache_system: bool = False,   # kept for API compatibility, no-op on Groq
        max_retries: int = 3,
    ) -> str:
        """Return raw text from the model with exponential-backoff retry."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self._get_client().chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=0.1,   # low temp → deterministic SQL / JSON
                )
                return response.choices[0].message.content
            except RateLimitError as exc:
                wait = 2 ** attempt
                logger.warning("Groq rate limited — retrying in %ds (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
                last_error = exc
            except APIStatusError as exc:
                logger.error("Groq API error on attempt %d: %s", attempt + 1, exc)
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(1)

        raise RuntimeError(f"LLM request failed after {max_retries} retries: {last_error}")

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
            obj_match = re.search(r"\{[\s\S]*\}", text)
            if obj_match:
                try:
                    return json.loads(obj_match.group(0))
                except json.JSONDecodeError:
                    pass
            logger.error("Non-JSON response: %s", text[:300])
            raise ValueError(f"Model returned non-JSON output: {text[:300]}")


llm_client = LLMClient()

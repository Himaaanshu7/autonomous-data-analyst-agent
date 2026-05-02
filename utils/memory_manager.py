from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryManager:
    """Stores per-session query history, schema cache, and context."""

    max_history: int = 10
    _history: deque = field(default_factory=lambda: deque(maxlen=10))
    _schema_cache: str = field(default="")
    _context: dict[str, Any] = field(default_factory=dict)

    def add_query(self, query: str, result_summary: str) -> None:
        self._history.append({"query": query, "summary": result_summary})

    def get_history_text(self) -> str:
        if not self._history:
            return "No previous queries in this session."
        return "\n".join(
            f"[{i+1}] Q: {item['query']}\n     A: {item['summary']}"
            for i, item in enumerate(self._history)
        )

    def cache_schema(self, schema_text: str) -> None:
        self._schema_cache = schema_text

    def get_cached_schema(self) -> str:
        return self._schema_cache

    def set(self, key: str, value: Any) -> None:
        self._context[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._context.get(key, default)

    def clear_history(self) -> None:
        self._history.clear()

    def clear_all(self) -> None:
        self._history.clear()
        self._schema_cache = ""
        self._context.clear()


memory = MemoryManager()

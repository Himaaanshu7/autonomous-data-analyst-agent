import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HISTORY_FILE = Path(__file__).resolve().parent.parent / "data" / "memory_history.json"


@dataclass
class MemoryManager:
    """Stores per-session query history, schema cache, and context."""

    max_history: int = 10
    _history: deque = field(default_factory=lambda: deque(maxlen=10))
    _schema_cache: str = field(default="")
    _context: dict[str, Any] = field(default_factory=dict)

    def add_query(self, query: str, result_summary: str) -> None:
        self._history.append({"query": query, "summary": result_summary})
        self._save()

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
        try:
            _HISTORY_FILE.unlink(missing_ok=True)
        except Exception:
            pass

    def clear_all(self) -> None:
        self.clear_history()
        self._schema_cache = ""
        self._context.clear()

    def load(self) -> None:
        try:
            if _HISTORY_FILE.exists():
                data = json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
                self._history = deque(data, maxlen=self.max_history)
        except Exception as exc:
            logger.warning("Could not load memory history: %s", exc)

    def _save(self) -> None:
        try:
            _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            _HISTORY_FILE.write_text(
                json.dumps(list(self._history), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Could not save memory history: %s", exc)


memory = MemoryManager()
memory.load()

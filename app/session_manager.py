"""Named session save/load for the chat history."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SESSIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
    except ImportError:
        pass
    return str(obj)


def save_session(name: str, messages: list, active_tables: list) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip()
    if not safe_name:
        safe_name = "session"
    path = SESSIONS_DIR / f"{safe_name}.json"
    payload = {
        "name": name,
        "saved_at": datetime.now().isoformat(),
        "active_tables": active_tables,
        "messages": _slim_messages(messages),
    }
    path.write_text(json.dumps(payload, default=_json_default, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Session saved: %s", path)


def list_sessions() -> list[str]:
    if not SESSIONS_DIR.exists():
        return []
    return sorted(p.stem for p in SESSIONS_DIR.glob("*.json"))


def load_session(name: str) -> dict:
    path = SESSIONS_DIR / f"{name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def delete_session(name: str) -> None:
    path = SESSIONS_DIR / f"{name}.json"
    path.unlink(missing_ok=True)


def _slim_messages(messages: list) -> list:
    result = []
    for msg in messages:
        m: dict = {"role": msg["role"], "content": msg["content"]}
        if "report" in msg:
            r = dict(msg["report"])
            data = dict(r.get("data", {}))
            data["rows"] = data.get("rows", [])[:50]
            r["data"] = data
            m["report"] = r
        result.append(m)
    return result

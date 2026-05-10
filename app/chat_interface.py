"""Chat interface component — message history + spinner + sample prompts."""
import json
import logging
from pathlib import Path
from typing import Any, Callable

import streamlit as st

logger = logging.getLogger(__name__)

_CHAT_FILE = Path(__file__).resolve().parent.parent / "data" / "chat_history.json"


def _json_default(obj: Any) -> Any:
    """Fallback serialiser for numpy/pandas types."""
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)


def _slim_report(report: dict) -> dict:
    """Cap data rows so the history file stays small."""
    import copy
    r = copy.deepcopy(report)
    data = r.get("data", {})
    if isinstance(data, dict) and "rows" in data:
        data["rows"] = data["rows"][:50]
    return r


def _save_chat(messages: list) -> None:
    try:
        _CHAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        to_save = []
        for msg in messages:
            m: dict = {"role": msg["role"], "content": msg["content"]}
            if "report" in msg:
                m["report"] = _slim_report(msg["report"])
            to_save.append(m)
        _CHAT_FILE.write_text(
            json.dumps(to_save, default=_json_default, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Could not save chat history: %s", exc)


def load_chat_history() -> list:
    try:
        if _CHAT_FILE.exists():
            return json.loads(_CHAT_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load chat history: %s", exc)
    return []


SAMPLE_QUERIES = [
    "Why did revenue drop last month?",
    "Show me top 5 customers by total revenue",
    "Detect anomalies in sales data",
    "What is the revenue trend over the last 12 months?",
    "Which region has the highest profit margin?",
    "Show correlation between salary and performance rating",
    "Which product category has the most returns?",
    "What is the attrition rate by department?",
]


def render_chat_history() -> None:
    """Render all messages stored in st.session_state.messages."""
    for msg in st.session_state.get("messages", []):
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])

            # If this assistant message has a report attached, render it
            if role == "assistant" and "report" in msg:
                from app.dashboard import render_report
                render_report(msg["report"])


def render_sample_queries(on_select: Callable[[str], None]) -> None:
    """Show clickable sample queries in the sidebar."""
    st.sidebar.subheader("Sample Queries")
    for q in SAMPLE_QUERIES:
        if st.sidebar.button(q, key=f"sample_{q[:20]}", use_container_width=True):
            on_select(q)


def add_user_message(content: str) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": "user", "content": content})
    _save_chat(st.session_state.messages)


def add_assistant_message(content: str, report: dict | None = None) -> None:
    msg: dict = {"role": "assistant", "content": content}
    if report:
        msg["report"] = report
    st.session_state.messages.append(msg)
    _save_chat(st.session_state.messages)


def clear_chat() -> None:
    st.session_state.messages = []
    try:
        _CHAT_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    from utils.memory_manager import memory
    memory.clear_history()

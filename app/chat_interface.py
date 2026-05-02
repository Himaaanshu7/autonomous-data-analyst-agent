"""Chat interface component — message history + spinner + sample prompts."""
import streamlit as st
from typing import Callable


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


def add_assistant_message(content: str, report: dict | None = None) -> None:
    msg: dict = {"role": "assistant", "content": content}
    if report:
        msg["report"] = report
    st.session_state.messages.append(msg)


def clear_chat() -> None:
    st.session_state.messages = []
    from utils.memory_manager import memory
    memory.clear_history()

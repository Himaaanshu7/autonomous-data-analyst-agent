"""Streamlit entry point for the Autonomous Data Analyst Agent."""
import streamlit as st
import logging
import sys
from pathlib import Path

# Ensure project root is on the path when launched from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

# ── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="Autonomous Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.chat_interface import (
    render_chat_history,
    render_sample_queries,
    add_user_message,
    add_assistant_message,
    clear_chat,
)
from app.dashboard import render_autonomous_alerts
from utils.db_connector import load_all_sample_data, list_tables, load_csv_as_table
from utils.schema_inspector import get_all_schemas_as_text
from utils.memory_manager import memory


def _check_api_key() -> bool:
    """Return True if a valid Groq key is present; otherwise render setup instructions."""
    from config.settings import settings
    key = settings.groq_api_key
    if key and key.startswith("gsk_") and "your-key-here" not in key:
        return True

    st.error("**Groq API key not configured.**")
    st.markdown(
        """
**Groq is free — no billing required.**

**Step 1 — Get your free key:**
Go to https://console.groq.com/keys → sign up → create an API key.

**Step 2 — Add it locally:**
Open `.env` in the project root and set:
```
GROQ_API_KEY=gsk_your-real-key-here
```

**Step 3 — Restart the app.**

---
**Deploying on Streamlit Cloud?**
Add `GROQ_API_KEY = "gsk_..."` to your app's **Secrets** panel.
"""
    )
    return False


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "autonomous_alerts" not in st.session_state:
        st.session_state.autonomous_alerts = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    st.sidebar.title("🤖 Data Analyst Agent")
    st.sidebar.markdown("---")

    # ── Dataset Management ─────────────────────────────────────────────
    st.sidebar.subheader("Dataset")
    col_load, col_clear = st.sidebar.columns(2)

    if col_load.button("Load Sample Data", use_container_width=True):
        with st.spinner("Loading datasets into DuckDB…"):
            tables = load_all_sample_data()
            schema = get_all_schemas_as_text()
            memory.cache_schema(schema)
            st.session_state.data_loaded = True
        st.sidebar.success(f"Loaded: {', '.join(tables)}")
        st.rerun()

    if col_clear.button("Clear Chat", use_container_width=True):
        clear_chat()
        st.rerun()

    # ── CSV Upload ─────────────────────────────────────────────────────
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        table_name = Path(uploaded.name).stem.lower().replace(" ", "_")
        load_csv_as_table(tmp_path, table_name, overwrite=True)
        os.unlink(tmp_path)
        schema = get_all_schemas_as_text()
        memory.cache_schema(schema)
        st.session_state.data_loaded = True
        st.sidebar.success(f"Loaded as table: '{table_name}'")
        st.rerun()

    # ── Loaded Tables ──────────────────────────────────────────────────
    tables = list_tables()
    if tables:
        st.sidebar.subheader("Loaded Tables")
        for t in tables:
            st.sidebar.markdown(f"- `{t}`")

    # ── Autonomous Scan ────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Autonomous Mode")
    if st.sidebar.button("Run Anomaly Scan", use_container_width=True, type="primary"):
        _run_autonomous_scan()

    # ── Sample Queries ─────────────────────────────────────────────────
    st.sidebar.markdown("---")
    render_sample_queries(on_select=_handle_sample_query)

    # ── Settings ──────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    with st.sidebar.expander("Settings"):
        from config.settings import settings
        st.text(f"Model: {settings.llm_model}")
        st.text(f"DB: {Path(settings.duckdb_path).name}")


# ---------------------------------------------------------------------------
# Autonomous Scan
# ---------------------------------------------------------------------------

def _run_autonomous_scan() -> None:
    from tools.anomaly_detector import detect_anomalies
    from utils.db_connector import run_query, list_tables as lt
    from agents.prompts import AUTONOMOUS_SCAN_PROMPT
    from utils.llm_client import llm_client
    import pandas as pd

    tables = lt()
    alerts = []

    with st.spinner("Running autonomous anomaly scan…"):
        for table in tables:
            try:
                df = run_query(f'SELECT * FROM "{table}" LIMIT 5000')
                result = detect_anomalies(df)
                if result.get("total_anomalies", 0) > 0:
                    alert_prompt = AUTONOMOUS_SCAN_PROMPT.format(
                        table_name=table,
                        anomaly_details=result.get("summary", ""),
                    )
                    alert_text = llm_client.complete(
                        system="You are a business data monitoring agent.",
                        user=alert_prompt,
                    )
                    alerts.append(
                        {
                            "table": table,
                            "anomaly_count": result["total_anomalies"],
                            "alert_text": alert_text,
                        }
                    )
            except Exception as exc:
                st.warning(f"Could not scan '{table}': {exc}")

    st.session_state.autonomous_alerts = alerts
    st.rerun()


# ---------------------------------------------------------------------------
# Handle incoming user message
# ---------------------------------------------------------------------------

def _handle_query(user_query: str) -> None:
    if not list_tables():
        st.warning("Please load a dataset first (sidebar → Load Sample Data).")
        return

    add_user_message(user_query)

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analysing…"):
            from agents.analyst_agent import run_query as agent_run
            try:
                report = agent_run(user_query)
            except Exception as exc:
                logging.exception("Agent error")
                report = {
                    "status": "error",
                    "error": str(exc),
                    "executive_summary": f"An error occurred: {exc}",
                    "sql": "",
                    "data": {},
                    "key_findings": [],
                    "recommendations": [],
                    "visualizations": [],
                }

        summary = report.get("executive_summary", "Analysis complete.")
        st.markdown(summary)
        from app.dashboard import render_report
        render_report(report)

    add_assistant_message(summary, report=report)


def _handle_sample_query(query: str) -> None:
    st.session_state["_pending_query"] = query
    st.rerun()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session()

    if not _check_api_key():
        st.stop()

    # Auto-load schema into memory if tables already exist
    if not memory.get_cached_schema() and list_tables():
        memory.cache_schema(get_all_schemas_as_text())
        st.session_state.data_loaded = True

    render_sidebar()

    # ── Header ─────────────────────────────────────────────────────────
    st.title("🤖 Autonomous Data Analyst Agent")
    st.markdown(
        "Ask any question about your data in plain English. "
        "The agent will write SQL, analyse results, detect anomalies, "
        "and generate business insights automatically."
    )

    # ── Autonomous Alerts ──────────────────────────────────────────────
    if st.session_state.autonomous_alerts:
        render_autonomous_alerts(st.session_state.autonomous_alerts)
        st.divider()

    # ── Chat ───────────────────────────────────────────────────────────
    render_chat_history()

    # Process pending sample query (set by sidebar button)
    pending = st.session_state.pop("_pending_query", None)
    if pending:
        _handle_query(pending)

    # Chat input
    user_input = st.chat_input("Ask a question about your data…")
    if user_input:
        _handle_query(user_input)


if __name__ == "__main__":
    main()

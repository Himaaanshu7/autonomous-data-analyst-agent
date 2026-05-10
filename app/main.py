"""Streamlit entry point for Noctua — AI Data Analyst."""
import logging
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Noctua — AI Data Analyst",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.chat_interface import (
    render_chat_history,
    render_sample_queries,
    add_user_message,
    add_assistant_message,
    clear_chat,
    load_chat_history,
)
from app.dashboard import render_autonomous_alerts
from utils.db_connector import (
    load_all_sample_data, list_tables, load_csv_as_table, drop_table,
    execute_statement, create_filtered_view, drop_view, list_views, run_query,
)
from utils.schema_inspector import get_all_schemas_as_text, get_schema_for_tables
from utils.memory_manager import memory


# ---------------------------------------------------------------------------
# Dark mode CSS
# ---------------------------------------------------------------------------

_DARK_CSS = """
<style>
[data-testid="stAppViewContainer"] { background-color: #0e1117 !important; color: #fafafa !important; }
[data-testid="stSidebar"] { background-color: #161b22 !important; }
[data-testid="stSidebar"] * { color: #fafafa !important; }
[data-testid="stChatMessage"] { background-color: #1c2128 !important; }
[data-testid="stExpander"] { background-color: #1c2128 !important; }
[data-testid="stMetric"] { background-color: #1c2128 !important; border-radius: 8px; padding: 8px; }
.stMarkdown, .stText, p, h1, h2, h3, h4, label { color: #fafafa !important; }
[data-testid="stDataFrame"] { background-color: #1c2128 !important; }
.stTabs [data-baseweb="tab"] { background-color: #1c2128 !important; color: #fafafa !important; }
</style>
"""


def _inject_dark_mode_css() -> None:
    if st.session_state.get("dark_mode"):
        st.markdown(_DARK_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API key check
# ---------------------------------------------------------------------------

def _check_api_key() -> bool:
    import os
    key = ""
    # Try Streamlit secrets first (cloud deploy)
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            os.environ["GROQ_API_KEY"] = key
    except Exception:
        pass
    # Fall back to env / .env
    if not key:
        from config.settings import settings
        key = settings.groq_api_key
    if key and key.startswith("gsk_") and "your-key-here" not in key:
        return True
    st.error("**Groq API key not configured.**")
    st.markdown("""
**Groq is free — no billing required.**

**Step 1 — Get your free key:**
Go to https://console.groq.com/keys → sign up → create an API key.

**Step 2 — Add it locally:**
Open `.env` in the project root and set:
```
GROQ_API_KEY=gsk_your-real-key-here
```
**Step 3 — Restart the app.**
""")
    return False


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()
    defaults = {
        "data_loaded": False,
        "autonomous_alerts": [],
        "active_tables": [],
        "deleted_tables": set(),
        "saved_queries": [],
        "suggested_questions": [],
        "threshold_alerts": [],
        "triggered_alerts": [],
        "dark_mode": False,
        "session_name": "",
        "conv_pdf_ready": False,
        "conv_pdf_bytes": b"",
        "filter_active": False,
        "filter_view_name": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Helper: load sample data (shared by sidebar button + welcome screen)
# ---------------------------------------------------------------------------

def _do_load_sample_data() -> None:
    with st.spinner("Loading datasets into DuckDB…"):
        tables = load_all_sample_data()
        tables = [t for t in tables if t not in st.session_state.deleted_tables]
        for t in list(st.session_state.deleted_tables):
            drop_table(t)
        st.session_state.active_tables = tables
        schema = get_schema_for_tables(tables)
        memory.cache_schema(schema)
        st.session_state.data_loaded = True
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar sections
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    st.sidebar.markdown("""
<div style="
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 14px;
    padding: 18px 16px 14px 16px;
    margin-bottom: 6px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.4);
    text-align: center;
">
    <div style="font-size: 2.4rem; line-height: 1;">🦉</div>
    <div style="
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        background: linear-gradient(90deg, #a8edea, #fed6e3, #a8edea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-top: 4px;
    ">NOCTUA</div>
    <div style="
        font-size: 0.7rem;
        color: #8ecae6;
        letter-spacing: 0.18em;
        margin-top: 2px;
        text-transform: uppercase;
    ">AI Data Analyst</div>
</div>
""", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # ── Dataset Management ─────────────────────────────────────────────
    st.sidebar.subheader("Dataset")
    col_load, col_clear = st.sidebar.columns(2)

    if col_load.button("Load Sample Data", use_container_width=True):
        _do_load_sample_data()

    if col_clear.button("Clear Chat", use_container_width=True):
        clear_chat()
        st.rerun()

    # ── CSV Upload ─────────────────────────────────────────────────────
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded and uploaded.name != st.session_state.get("_last_upload"):
        import tempfile, os
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv")
        try:
            with os.fdopen(tmp_fd, "wb") as tmp:
                tmp.write(uploaded.getvalue())
            table_name = Path(uploaded.name).stem.lower().replace(" ", "_").replace("-", "_")
            load_csv_as_table(tmp_path, table_name, overwrite=True)
            st.session_state.active_tables = [table_name]
            schema = get_schema_for_tables([table_name])
            memory.cache_schema(schema)
            memory.clear_history()
            st.session_state.data_loaded = True
            st.session_state["_last_upload"] = uploaded.name
            st.sidebar.success(f"Loaded: '{table_name}'")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Could not load file: {exc}")
            logger.exception("CSV upload failed")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ── Table Manager ──────────────────────────────────────────────────
    tables = list_tables()
    if tables:
        st.sidebar.subheader("Loaded Tables")
        st.sidebar.caption("✅ = query this table   🗑 = delete it")
        active = st.session_state.active_tables or tables
        new_active = []

        for t in tables:
            col_check, col_del = st.sidebar.columns([5, 1])
            checked = col_check.checkbox(f"`{t}`", value=(t in active), key=f"tbl_{t}")
            if checked:
                new_active.append(t)
            if col_del.button("🗑", key=f"del_{t}", help=f"Delete table '{t}'"):
                drop_table(t)
                st.session_state.deleted_tables.add(t)
                st.session_state.active_tables = [x for x in st.session_state.active_tables if x != t]
                st.sidebar.success(f"Deleted '{t}'")
                st.rerun()

        if new_active != st.session_state.active_tables:
            st.session_state.active_tables = new_active
            memory.cache_schema(get_schema_for_tables(new_active))
            st.rerun()

    # ── Column Profiler ────────────────────────────────────────────────
    all_tables = list_tables()
    if all_tables:
        st.sidebar.markdown("---")
        with st.sidebar.expander("🔍 Column Profiler"):
            p_table = st.selectbox("Table", all_tables, key="profiler_table")
            if p_table:
                try:
                    p_df = run_query(f'SELECT * FROM "{p_table}" LIMIT 5000')
                    p_col = st.selectbox("Column", p_df.columns.tolist(), key="profiler_col")
                    if p_col:
                        s = p_df[p_col]
                        null_count = int(s.isna().sum())
                        null_pct = round(null_count / max(len(s), 1) * 100, 1)
                        st.metric("Null %", f"{null_pct}%")
                        if s.dtype.kind in "iuf":
                            c1, c2 = st.columns(2)
                            c1.metric("Min", f"{s.min():,.2f}")
                            c2.metric("Max", f"{s.max():,.2f}")
                            c3, c4 = st.columns(2)
                            c3.metric("Mean", f"{s.mean():,.2f}")
                            c4.metric("Std Dev", f"{s.std():,.2f}")
                        else:
                            top = s.value_counts().head(5)
                            st.markdown("**Top values:**")
                            for val, cnt in top.items():
                                st.caption(f"`{val}` — {cnt:,} times")
                except Exception as exc:
                    st.warning(f"Could not profile: {exc}")

    # ── Data Filters ───────────────────────────────────────────────────
    if all_tables:
        with st.sidebar.expander("🎚️ Data Filters"):
            f_table = st.selectbox("Filter table", all_tables, key="filter_table_select")
            if f_table:
                try:
                    f_df = run_query(f'SELECT * FROM "{f_table}" LIMIT 5000')
                    date_cols = [c for c in f_df.columns if any(k in c.lower() for k in ("date", "time", "month", "year"))]
                    cat_cols = [c for c in f_df.columns if f_df[c].dtype == object and f_df[c].nunique() < 50]

                    where_parts = []
                    for dc in date_cols[:1]:
                        try:
                            dates = sorted(f_df[dc].dropna().unique())
                            min_d = str(dates[0])[:10]
                            max_d = str(dates[-1])[:10]
                            picked = st.date_input(f"Date range ({dc})", key=f"f_date_{f_table}_{dc}")
                            if isinstance(picked, (list, tuple)) and len(picked) == 2:
                                where_parts.append(f'CAST("{dc}" AS DATE) BETWEEN \'{picked[0]}\' AND \'{picked[1]}\'')
                        except Exception:
                            pass

                    for cc in cat_cols[:3]:
                        opts = sorted(f_df[cc].dropna().unique().tolist())
                        sel = st.multiselect(f"{cc.replace('_',' ').title()}", opts, key=f"f_cat_{f_table}_{cc}")
                        if sel:
                            vals = ", ".join(f"'{v}'" for v in sel)
                            where_parts.append(f'"{cc}" IN ({vals})')

                    col_apply, col_clear_f = st.columns(2)
                    if col_apply.button("Apply", key=f"apply_filter_{f_table}", use_container_width=True):
                        if where_parts:
                            view_name = create_filtered_view(f_table, " AND ".join(where_parts))
                            st.session_state.filter_view_name = view_name
                            st.session_state.filter_active = True
                            active = st.session_state.active_tables or all_tables
                            if view_name not in active:
                                active = active + [view_name]
                            st.session_state.active_tables = active
                            memory.cache_schema(get_schema_for_tables(active))
                            st.sidebar.success(f"Filter applied → `{view_name}`")
                            st.rerun()
                        else:
                            st.info("Set at least one filter first.")

                    if col_clear_f.button("Clear", key=f"clear_filter_{f_table}", use_container_width=True):
                        if st.session_state.filter_view_name:
                            drop_view(st.session_state.filter_view_name)
                            st.session_state.active_tables = [
                                t for t in st.session_state.active_tables
                                if t != st.session_state.filter_view_name
                            ]
                            st.session_state.filter_view_name = ""
                            st.session_state.filter_active = False
                            memory.cache_schema(get_schema_for_tables(st.session_state.active_tables))
                            st.rerun()
                except Exception as exc:
                    st.warning(f"Filter error: {exc}")

    # ── Calculated Columns ─────────────────────────────────────────────
    if all_tables:
        with st.sidebar.expander("➕ Calculated Columns"):
            with st.form("calc_col_form"):
                calc_table = st.selectbox("Table", all_tables, key="calc_table")
                new_col = st.text_input("New column name", placeholder="profit")
                expr = st.text_area("SQL expression", placeholder="price * quantity", height=68)
                submitted = st.form_submit_button("Add Column")
            if submitted:
                if calc_table and new_col and expr:
                    try:
                        execute_statement(f'ALTER TABLE "{calc_table}" ADD COLUMN "{new_col}" DOUBLE')
                    except Exception:
                        try:
                            execute_statement(f'ALTER TABLE "{calc_table}" DROP COLUMN IF EXISTS "{new_col}"')
                            execute_statement(f'ALTER TABLE "{calc_table}" ADD COLUMN "{new_col}" DOUBLE')
                        except Exception as exc:
                            st.sidebar.error(f"Could not add column: {exc}")
                    try:
                        execute_statement(f'UPDATE "{calc_table}" SET "{new_col}" = {expr}')
                        active = st.session_state.active_tables or all_tables
                        memory.cache_schema(get_schema_for_tables(active))
                        st.sidebar.success(f"Column '{new_col}' added to '{calc_table}'!")
                        st.rerun()
                    except Exception as exc:
                        st.sidebar.error(f"Expression error: {exc}")
                else:
                    st.sidebar.warning("Fill in all fields.")

    # ── Threshold Alerts ───────────────────────────────────────────────
    if all_tables:
        with st.sidebar.expander("🔔 Threshold Alerts"):
            with st.form("alert_form"):
                a_table = st.selectbox("Table", all_tables, key="alert_table")
                a_col = st.text_input("Column name", placeholder="total")
                a_op = st.selectbox("Operator", [">", "<", ">=", "<=", "==", "!="])
                a_val = st.number_input("Threshold value", value=0.0)
                a_label = st.text_input("Alert label", placeholder="Revenue below target")
                add_alert = st.form_submit_button("Add Alert")
            if add_alert and a_col:
                st.session_state.threshold_alerts.append(
                    {"table": a_table, "col": a_col, "op": a_op, "val": a_val,
                     "label": a_label or f"{a_col} {a_op} {a_val}"}
                )
                st.sidebar.success("Alert added!")

            alerts = st.session_state.threshold_alerts
            if alerts:
                st.markdown("**Active alerts:**")
                for idx, alert in enumerate(alerts):
                    c1, c2 = st.sidebar.columns([5, 1])
                    c1.caption(f"{alert['label']}")
                    if c2.button("✕", key=f"del_alert_{idx}"):
                        st.session_state.threshold_alerts.pop(idx)
                        st.rerun()

    # ── Sessions ───────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    with st.sidebar.expander("💾 Sessions"):
        from app.session_manager import save_session, list_sessions, load_session
        from app.chat_interface import _save_chat

        session_name = st.text_input(
            "Session name",
            value=st.session_state.get("session_name", ""),
            key="session_name_input",
            placeholder="My Analysis",
        )
        col_save, col_new = st.sidebar.columns(2)
        if col_save.button("Save", key="save_session_btn", use_container_width=True):
            name = session_name.strip() or f"Session {datetime.now().strftime('%m-%d %H:%M')}"
            save_session(name, st.session_state.messages, st.session_state.active_tables)
            st.session_state.session_name = name
            st.sidebar.success(f"Saved: {name}")

        sessions = list_sessions()
        if sessions:
            load_sel = st.selectbox("Load session", [""] + sessions, key="session_load_select")
            if st.button("Load", key="load_session_btn", use_container_width=True) and load_sel:
                data = load_session(load_sel)
                msgs = data.get("messages", [])
                st.session_state.messages = msgs
                _save_chat(msgs)
                loaded_tables = data.get("active_tables", [])
                if loaded_tables:
                    st.session_state.active_tables = loaded_tables
                    memory.cache_schema(get_schema_for_tables(loaded_tables))
                st.session_state.session_name = load_sel
                st.rerun()

    # ── Autonomous Scan ────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Autonomous Mode")
    if st.sidebar.button("Run Anomaly Scan", use_container_width=True, type="primary"):
        _run_autonomous_scan()

    # ── Sample Queries ─────────────────────────────────────────────────
    st.sidebar.markdown("---")
    render_sample_queries(on_select=_handle_sample_query)

    # ── Watermark ─────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='text-align:center; color:#888; font-size:0.75rem;'>Built by <strong>Himanshu Mishra</strong></div>",
        unsafe_allow_html=True,
    )

    # ── Settings ──────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚙️ Settings"):
        from config.settings import settings
        st.text(f"Model: {settings.llm_model}")
        st.text(f"DB: {Path(settings.duckdb_path).name}")

        dark = st.checkbox("Dark mode", value=st.session_state.get("dark_mode", False), key="dark_mode_toggle")
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()

        st.markdown("---")
        # Export full conversation PDF
        if st.button("Export Chat PDF", key="export_chat_pdf_btn", use_container_width=True):
            if st.session_state.messages:
                from tools.pdf_exporter import export_conversation_to_pdf
                st.session_state.conv_pdf_bytes = export_conversation_to_pdf(st.session_state.messages)
                st.session_state.conv_pdf_ready = True
                st.rerun()
            else:
                st.warning("No messages to export.")

        if st.session_state.conv_pdf_ready and st.session_state.conv_pdf_bytes:
            st.download_button(
                "⬇️ Download Chat PDF",
                data=st.session_state.conv_pdf_bytes,
                file_name="conversation.pdf",
                mime="application/pdf",
                key="dl_conv_pdf",
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Autonomous Scan
# ---------------------------------------------------------------------------

def _run_autonomous_scan() -> None:
    from tools.anomaly_detector import detect_anomalies
    from agents.prompts import AUTONOMOUS_SCAN_PROMPT
    from utils.llm_client import llm_client

    tables = list_tables()
    alerts = []
    with st.spinner("Running autonomous anomaly scan…"):
        for table in tables:
            try:
                df = run_query(f'SELECT * FROM "{table}" LIMIT 5000')
                result = detect_anomalies(df)
                if result.get("total_anomalies", 0) > 0:
                    alert_text = llm_client.complete(
                        system="You are a business data monitoring agent.",
                        user=AUTONOMOUS_SCAN_PROMPT.format(
                            table_name=table,
                            anomaly_details=result.get("summary", ""),
                        ),
                    )
                    alerts.append({"table": table, "anomaly_count": result["total_anomalies"], "alert_text": alert_text})
            except Exception as exc:
                st.warning(f"Could not scan '{table}': {exc}")

    st.session_state.autonomous_alerts = alerts
    st.rerun()


# ---------------------------------------------------------------------------
# Threshold alert checker
# ---------------------------------------------------------------------------

def _check_threshold_alerts() -> None:
    from utils.db_connector import get_connection
    triggered = []
    for alert in st.session_state.get("threshold_alerts", []):
        try:
            op_map = {"==": "="}
            op = op_map.get(alert["op"], alert["op"])
            result = get_connection().execute(
                f'SELECT count(*) FROM "{alert["table"]}" WHERE "{alert["col"]}" {op} {alert["val"]}'
            ).fetchone()
            count = result[0] if result else 0
            if count > 0:
                triggered.append(f"🔔 **{alert['label']}** — {count:,} records match `{alert['col']} {alert['op']} {alert['val']}`")
        except Exception:
            pass
    st.session_state.triggered_alerts = triggered


# ---------------------------------------------------------------------------
# Welcome / onboarding screen
# ---------------------------------------------------------------------------

def _render_welcome_screen() -> None:
    st.markdown("---")
    st.markdown("## 👋 Welcome to Noctua")
    st.markdown("Ask questions about your data in plain English — no SQL or coding needed.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1 — Load Data**\nClick **Load Sample Data** in the sidebar, or upload your own CSV file.")
    with col2:
        st.info("**Step 2 — Ask a Question**\nType anything: *'Show me total sales by category'* or *'Find anomalies'*")
    with col3:
        st.info("**Step 3 — Explore Results**\nGet charts, insights, recommendations, and export reports.")

    st.markdown("---")
    st.markdown("**Or get started instantly with sample data:**")
    if st.button("🚀 Load Sample Datasets & Start", type="primary", use_container_width=False):
        _do_load_sample_data()


# ---------------------------------------------------------------------------
# Handle incoming user message
# ---------------------------------------------------------------------------

def _handle_query(user_query: str) -> None:
    if not list_tables():
        st.warning("Please load a dataset first (sidebar → Load Sample Data).")
        return

    st.session_state.triggered_alerts = []
    add_user_message(user_query)

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analysing…"):
            from agents.analyst_agent import run_query as agent_run
            active = st.session_state.get("active_tables") or list_tables()
            schema = get_schema_for_tables(active)
            try:
                report = agent_run(user_query, schema_text=schema)
            except Exception as exc:
                logger.exception("Agent error")
                report = {
                    "status": "error", "error": str(exc),
                    "executive_summary": f"An error occurred: {exc}",
                    "sql": "", "data": {}, "key_findings": [],
                    "recommendations": [], "visualizations": [],
                }

        summary = report.get("executive_summary", "Analysis complete.")
        st.markdown(summary)
        from app.dashboard import render_report
        render_report(report)

    add_assistant_message(summary, report=report)
    _check_threshold_alerts()


def _handle_sample_query(query: str) -> None:
    st.session_state["_pending_query"] = query
    st.rerun()


def _render_suggested_questions(active_tables: list) -> None:
    if "suggested_questions" not in st.session_state:
        schema = get_schema_for_tables(active_tables)
        with st.spinner("Generating suggested questions…"):
            from tools.question_suggester import suggest_questions
            st.session_state.suggested_questions = suggest_questions(schema, n=6)

    questions = st.session_state.get("suggested_questions", [])
    if not questions:
        return

    st.markdown("**Suggested questions for your dataset:**")
    cols = st.columns(2)
    for i, q in enumerate(questions):
        if cols[i % 2].button(q, key=f"sq_{i}", use_container_width=True):
            st.session_state.suggested_questions = []
            _handle_query(q)


def _render_saved_queries() -> None:
    saved: list = st.session_state.get("saved_queries", [])
    st.subheader("Saved Queries")
    if not saved:
        st.info("No saved queries yet. After running a query, click **Save this query** to bookmark it.")
        return
    for i, item in enumerate(saved):
        col_q, col_run, col_del = st.columns([6, 1, 1])
        col_q.markdown(f"**{i+1}.** {item['query']}")
        col_q.caption(item.get("saved_at", ""))
        if col_run.button("▶", key=f"run_saved_{i}", help="Re-run this query"):
            _handle_query(item["query"])
        if col_del.button("🗑", key=f"del_saved_{i}", help="Delete this saved query"):
            st.session_state.saved_queries.pop(i)
            st.rerun()


def _save_current_query(query: str) -> None:
    if "saved_queries" not in st.session_state:
        st.session_state.saved_queries = []
    existing = [s["query"] for s in st.session_state.saved_queries]
    if query not in existing:
        st.session_state.saved_queries.append({
            "query": query,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session()
    _inject_dark_mode_css()

    if not _check_api_key():
        st.stop()

    for t in list(st.session_state.deleted_tables):
        drop_table(t)

    all_tables = [t for t in list_tables() if t not in st.session_state.deleted_tables]
    if all_tables and not st.session_state.active_tables:
        st.session_state.active_tables = all_tables
    if not memory.get_cached_schema() and all_tables:
        active = [t for t in (st.session_state.active_tables or all_tables)
                  if t not in st.session_state.deleted_tables]
        memory.cache_schema(get_schema_for_tables(active))
        st.session_state.data_loaded = True

    render_sidebar()

    st.markdown("""
<div style="
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 12px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.35);
">
    <div style="display:flex; align-items:center; gap:16px;">
        <span style="font-size:3rem;">🦉</span>
        <div>
            <div style="
                font-size: 2.4rem;
                font-weight: 900;
                letter-spacing: 0.1em;
                background: linear-gradient(90deg, #a8edea, #fed6e3);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                line-height: 1.1;
            ">NOCTUA</div>
            <div style="color:#8ecae6; font-size:0.85rem; letter-spacing:0.2em; text-transform:uppercase; margin-top:2px;">
                See everything in your data
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Threshold alert banners (top of page)
    for alert_msg in st.session_state.get("triggered_alerts", []):
        st.warning(alert_msg)

    user_input = st.chat_input("Ask a question about your data…")

    tab_chat, tab_kpi, tab_saved = st.tabs(["💬 Chat", "📊 KPI Dashboard", "🔖 Saved Queries"])

    with tab_chat:
        if st.session_state.autonomous_alerts:
            render_autonomous_alerts(st.session_state.autonomous_alerts)
            st.divider()

        active = st.session_state.get("active_tables", [])

        # Welcome screen when no data and no chat history
        if not active and not st.session_state.messages:
            _render_welcome_screen()
        else:
            if active and not st.session_state.messages:
                _render_suggested_questions(active)

            render_chat_history()

            pending = st.session_state.pop("_pending_query", None)
            if pending:
                _handle_query(pending)

            if user_input:
                _handle_query(user_input)

    with tab_kpi:
        from app.kpi_dashboard import render_kpi_dashboard
        active_tables = st.session_state.get("active_tables", [])
        if active_tables:
            if st.button("🔄 Refresh Dashboard", key="refresh_kpi"):
                st.session_state.pop("kpi_cache", None)
            render_kpi_dashboard(active_tables)
        else:
            st.info("Load a dataset first to see the KPI Dashboard.")

    with tab_saved:
        _render_saved_queries()


if __name__ == "__main__":
    main()

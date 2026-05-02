# Autonomous Data Analyst Agent

A production-grade AI agent that mimics a real data analyst — querying databases, detecting anomalies, generating visualizations, and producing actionable business insights from natural language queries.

## Architecture

```
User Query → LangGraph Agent → [SQL Gen → Executor → Profiler → Anomaly Detector → Viz → Insights] → Report
```

The agent uses a stateful graph with a reflection loop: if SQL execution fails, it automatically diagnoses the error and retries (up to 3 times) before returning a graceful error.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Framework | LangGraph 0.2+ |
| LLM | Claude Sonnet 4.6 (Anthropic) |
| Database | DuckDB (local columnar analytics) |
| Frontend | Streamlit |
| Visualization | Plotly |
| Data Processing | Pandas, NumPy, SciPy |

## Project Structure

```
autonomous-data-analyst-agent/
├── app/
│   ├── main.py              # Streamlit entry point
│   ├── chat_interface.py    # Chat UI component
│   └── dashboard.py         # Chart + report renderer
├── agents/
│   ├── analyst_agent.py     # LangGraph StateGraph (7 nodes)
│   ├── state.py             # AgentState TypedDict
│   └── prompts.py           # All prompt templates
├── tools/
│   ├── sql_generator.py     # NL → DuckDB SQL
│   ├── sql_executor.py      # SQL execution + error handling
│   ├── data_profiler.py     # Column stats, distributions
│   ├── anomaly_detector.py  # IQR + Z-score detection
│   ├── trend_detector.py    # MoM/YoY/moving avg
│   ├── correlation_analyzer.py
│   ├── viz_generator.py     # Plotly chart factory
│   ├── insight_generator.py # LLM-powered insights
│   └── report_generator.py  # Final report assembly
├── data/
│   ├── sample/              # Generated CSV datasets
│   └── generate_sample_data.py
├── utils/
│   ├── llm_client.py        # Anthropic API wrapper + retry
│   ├── db_connector.py      # DuckDB connection factory
│   ├── schema_inspector.py  # Table schema → text for prompts
│   └── memory_manager.py    # Query history + schema cache
├── config/
│   └── settings.py          # Pydantic BaseSettings
└── tests/
    └── test_tools.py
```

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd autonomous-data-analyst-agent
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Generate sample data

```bash
python data/generate_sample_data.py
```

### 4. Run the app

```bash
streamlit run app/main.py
```

Open http://localhost:8501 in your browser.

## Sample Queries

- "Why did revenue drop last month?"
- "Show me top 5 customers by total revenue"
- "Detect anomalies in sales data"
- "What is the revenue trend over the last 12 months?"
- "Which region has the highest profit margin?"
- "What is the attrition rate by department?"

## Running Tests

```bash
pytest tests/ -v
```

## Autonomous Mode

Click **Run Anomaly Scan** in the sidebar. The agent scans all loaded tables, detects statistical anomalies, and generates a business alert for each affected table — with no user query needed.

## Deployment (Streamlit Cloud)

1. Push to GitHub
2. Go to share.streamlit.io → New App
3. Set `app/main.py` as the entry point
4. Add `ANTHROPIC_API_KEY` in Secrets

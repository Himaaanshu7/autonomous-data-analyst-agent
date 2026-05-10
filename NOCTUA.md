# 🦉 Noctua — AI Data Analyst Agent

**Live Demo:** https://autonomous-data-analyst-agent.streamlit.app/

Noctua is an autonomous AI agent that thinks and acts like a real data analyst. Ask questions in plain English — Noctua queries your data, detects anomalies, forecasts trends, runs statistical tests, and delivers business-ready reports. No SQL, no coding required.

---

## What Noctua Can Do

| Capability | Description |
|---|---|
| Natural Language Querying | Ask anything — Noctua writes and runs the SQL |
| Anomaly Detection | Automatically flags statistical outliers in your data |
| Forecasting | Projects future values with confidence intervals |
| Cohort Analysis | Tracks customer/user retention over time |
| Statistical Testing | Compares groups using t-tests and ANOVA |
| Pivot Tables | Cross-tabulates any combination of columns |
| Data Cleaning | Fixes nulls, duplicates, type issues, and outliers |
| Data Validation | Checks business rules and data quality |
| KPI Dashboard | Live metrics and charts for any loaded dataset |
| Chart Generation | Auto-selects the right chart for your data |
| Export | Download reports as PDF or Excel |
| Session Management | Save and reload analysis sessions |

---

## Use Cases

### Business Analytics
- *"Why did revenue drop last month?"*
- *"Which region has the highest profit margin?"*
- *"Show me top 10 customers by total spend"*
- *"What is the month-over-month growth rate?"*

### Anomaly & Risk Monitoring
- *"Detect anomalies in sales data"*
- *"Flag any transactions above $50,000"*
- *"Are there any unusual spikes in refund rates?"*
- Run **Autonomous Anomaly Scan** — Noctua scans all tables with no query needed

### Forecasting & Trends
- *"Forecast revenue for the next 6 months"*
- *"What is the trend in customer acquisition over the last year?"*
- *"Show me a 3-month moving average of orders"*

### Customer & Retention Analysis
- *"Show me cohort retention for the last 12 months"*
- *"Which customer segment has the highest lifetime value?"*
- *"What is the churn rate by subscription tier?"*

### HR & Operations
- *"What is the attrition rate by department?"*
- *"Show headcount growth over the last 2 years"*
- *"Which team has the most open positions?"*

### Data Quality
- *"Clean my dataset and report what was fixed"*
- *"Validate that no sales amounts are negative"*
- *"How many missing values are in each column?"*

---

## How It Works

Noctua uses a **LangGraph multi-node agent** that routes every query through a series of intelligent steps:

```
User Query
    ↓
Planner         → decides what kind of analysis is needed
    ↓
Data Cleaner    → optional: fixes quality issues first
    ↓
SQL Generator   → writes DuckDB SQL from natural language
    ↓
SQL Executor    → runs the query (retries up to 3x if it fails)
    ↓
Analyzer        → profiles data, detects anomalies, trends, correlations
    ↓
Visualizer      → picks and builds the right chart automatically
    ↓
Insight Engine  → LLM generates findings and business recommendations
    ↓
Report Builder  → assembles everything into a structured report
```

Specialized paths kick in automatically based on query intent:
- **Forecast queries** → Forecaster node
- **Pivot/crosstab queries** → Pivot Analyzer node
- **Statistical comparison queries** → Statistical Tester node
- **Cohort/retention queries** → Cohort Analyzer node
- **Validation queries** → Validator node

---

## Tools & Capabilities in Detail

### Anomaly Detector
Detects statistical outliers in numeric columns using two methods:
- **IQR (Interquartile Range):** Flags values outside 1.5× the IQR bounds
- **Z-score:** Flags values more than 3 standard deviations from the mean
- Returns a count, percentage, and sample of anomalous rows per column

### Forecaster
Projects future values using linear regression:
- Auto-detects date and value columns
- Aggregates to monthly granularity
- Produces N-month forecasts (default: 6 months)
- Includes 95% confidence intervals
- Reports trend direction and projected % change

### Cohort Analyzer
Tracks how groups of users/customers behave over time:
- Groups users by their first activity month
- Calculates retention rate for each subsequent period (up to 12 months)
- Returns a full retention matrix and average retention by period
- Renders as a color-coded heatmap

### Statistical Tester
Runs significance tests to compare groups:
- **2 groups:** Welch's t-test (handles unequal variance)
- **3+ groups:** One-way ANOVA
- **Group balance:** Chi-square test
- Returns p-values, test statistics, and plain-English interpretations

### Pivot Analyzer
Builds cross-tabulation tables:
- Auto-selects row, column, and value fields
- Supports: sum, mean, count, max, min, median
- Renders as grouped bar chart

### Data Cleaner
Fixes common data quality issues automatically:
- Renames columns to `snake_case`
- Strips whitespace from text
- Infers correct data types
- Removes duplicate rows
- Fills missing values (numeric → median, categorical → mode)
- Drops columns with >60% null values
- Caps outliers using IQR bounds
- Returns a full report of every change made

### Validator
Checks business rules and data integrity:
- **Built-in rules:** no negative numerics, no future dates, no duplicates, high null rate detection
- **Custom rules:** define your own conditions (e.g. `amount > 0`, `status != null`)
- Returns a quality score (0–100) and per-rule violation details

### Chart Generator
Automatically selects and builds the right chart:
- Time series → Line chart with optional moving average
- Category comparison → Bar chart
- Two numeric columns → Scatter plot
- Correlation matrix → Heatmap
- Single distribution → Histogram
- Anomalies → Scatter with outliers highlighted in red
- Small results → Interactive table

### KPI Dashboard
Always-on metrics view for any loaded dataset:
- Row count, column count, null %, duplicate count
- Anomaly indicator with alert styling
- Trend direction with color-coded arrows (↑ green, ↓ red, → orange)
- Distribution histogram of key numeric column
- Top values bar chart for categorical columns

### Report Exporter
Download your analysis in multiple formats:
- **PDF:** Branded report with executive summary, findings, recommendations, anomaly details, and SQL query
- **Excel:** Multi-sheet workbook (Summary, Data, Insights, Anomalies, Trends, Correlations, SQL)
- **Conversation PDF:** Export the full chat history as a formatted PDF

### Question Suggester
When you load a dataset, Noctua reads the schema and auto-generates 6 practical questions tailored to your specific columns — so you always know where to start.

### Session Manager
Save and restore your entire analysis session:
- Saves chat history + active tables under a custom name
- Reload any past session to pick up where you left off

---

## Supported Data Sources

- **Sample datasets** — built-in (load with one click)
- **CSV upload** — drag and drop any CSV file
- **DuckDB** — queries run on an in-memory columnar database for fast analytics

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM | Groq (Llama 3.3 70B — free tier) |
| Database | DuckDB |
| Frontend | Streamlit |
| Visualization | Plotly |
| Data Processing | Pandas, NumPy, SciPy, scikit-learn |
| PDF Export | fpdf2 |
| Excel Export | openpyxl |

---

## Built by Himanshu Mishra

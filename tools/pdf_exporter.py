"""Export analysis report or full chat conversation to PDF using fpdf2."""
import io
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class _ReportPDF:
    """Thin wrapper so both export functions share the same header/footer."""

    def __new__(cls):
        from fpdf import FPDF

        class _PDF(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 11)
                self.set_fill_color(30, 80, 160)
                self.set_text_color(255, 255, 255)
                self.cell(0, 10, "  Autonomous Data Analyst", fill=True, ln=True)
                self.set_text_color(0, 0, 0)
                self.ln(2)

            def footer(self):
                self.set_y(-12)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(130, 130, 130)
                self.cell(0, 8, f"Page {self.page_no()} | {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

        pdf = _PDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_margins(15, 15, 15)
        pdf.add_page()
        return pdf


def _h1(pdf, text: str) -> None:
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_fill_color(240, 245, 255)
    pdf.cell(0, 8, text, fill=True, ln=True)
    pdf.ln(1)


def _h2(pdf, text: str) -> None:
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 80, 160)
    pdf.cell(0, 7, text, ln=True)
    pdf.set_text_color(0, 0, 0)


def _body(pdf, text: str, indent: int = 0) -> None:
    pdf.set_font("Helvetica", size=10)
    pdf.set_x(15 + indent)
    pdf.multi_cell(0, 6, str(text)[:600])


def _divider(pdf) -> None:
    pdf.set_draw_color(200, 200, 200)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)


def export_to_pdf(report: dict[str, Any]) -> bytes:
    """Build a formatted PDF from a single report dict. Returns bytes."""
    pdf = _ReportPDF()

    # Title block
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Data Analysis Report", ln=True, align="C")
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Query: {report.get('query', '')[:100]}", ln=True, align="C")
    pdf.cell(0, 6, f"Generated: {report.get('generated_at', '')}", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)
    _divider(pdf)

    _h1(pdf, "Executive Summary")
    _body(pdf, report.get("executive_summary", "N/A"))
    pdf.ln(3)

    _h1(pdf, "Key Metrics")
    data = report.get("data", {})
    trend = report.get("trends", {})
    for label, val in [
        ("Rows Returned", f"{data.get('row_count', 0):,}"),
        ("Anomalies Detected", str(report.get("anomaly_count", 0))),
        ("Trend Direction", trend.get("direction", "N/A").title()),
    ]:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(60, 7, label + ":")
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 7, val, ln=True)
    pdf.ln(3)

    findings = report.get("key_findings", [])
    if findings:
        _h1(pdf, "Key Findings")
        for i, f in enumerate(findings, 1):
            _body(pdf, f"  {i}. {f}", indent=3)
        pdf.ln(3)

    recs = report.get("recommendations", [])
    if recs:
        _h1(pdf, "Action Recommendations")
        for r in recs:
            priority = r.get("priority", "medium").upper()
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, f"  [{priority}] {r.get('action', '')}", ln=True)
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_x(20)
            pdf.multi_cell(0, 5, r.get("rationale", ""))
            pdf.ln(1)
        pdf.ln(2)

    anomaly_summary = report.get("anomalies", {}).get("summary", "")
    if anomaly_summary and report.get("anomaly_count", 0) > 0:
        _h1(pdf, "Anomaly Report")
        _body(pdf, anomaly_summary)
        pdf.ln(3)

    sql = report.get("sql", "")
    if sql:
        _h1(pdf, "SQL Query Used")
        pdf.set_font("Courier", size=8)
        pdf.set_fill_color(248, 248, 248)
        pdf.multi_cell(0, 5, sql[:800], fill=True)

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf.read()


def export_conversation_to_pdf(messages: list) -> bytes:
    """Export the full chat conversation to PDF. Returns bytes."""
    pdf = _ReportPDF()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Chat Conversation Export", ln=True, align="C")
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)
    _divider(pdf)

    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            pdf.set_fill_color(235, 245, 255)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, f"  You ({i}):", fill=True, ln=True)
            pdf.set_font("Helvetica", size=10)
            pdf.set_x(18)
            pdf.multi_cell(0, 6, content[:400])
            pdf.ln(2)
        else:
            pdf.set_fill_color(245, 255, 245)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, f"  Agent ({i}):", fill=True, ln=True)
            pdf.set_font("Helvetica", size=10)
            pdf.set_x(18)
            pdf.multi_cell(0, 6, content[:400])

            report = msg.get("report", {})
            findings = report.get("key_findings", [])
            if findings:
                pdf.set_font("Helvetica", "BI", 9)
                pdf.set_x(18)
                pdf.cell(0, 6, "Key Findings:", ln=True)
                pdf.set_font("Helvetica", size=9)
                for f in findings[:5]:
                    pdf.set_x(22)
                    pdf.multi_cell(0, 5, f"• {f[:200]}")
            pdf.ln(2)

        _divider(pdf)

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf.read()

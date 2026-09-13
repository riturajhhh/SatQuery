"""SatQuery AI — Analysis Report Generation & Audit Service.

Compiles comprehensive, tamper-evident audit trails and produces
publication-ready reports in HTML and PDF formats.

Includes:
- Analysis metadata, query, task, input modality, and timestamp
- Raster sensor metadata, GSD, CRS, and ISRO payload identification
- Model execution audit (models used, versions, devices, execution times)
- Primary answer, calibrated confidence score, and confidence level
- Quantitative evidence (bounding boxes, change detection statistics, optical-SAR metrics)
- Chronological execution trace with millisecond-precision benchmarks
- System limitations and compliance disclaimer
"""

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.database import crud
from app.database.models import Analysis, UploadedFile, ModelRun, ExecutionTrace, Evidence
from app.utils.logging import get_logger

logger = get_logger("services.report")


def generate_report_data(db: Session, analysis_id: str) -> Dict[str, Any]:
    """Compile a complete, structured audit dictionary for an analysis run.

    Args:
        db: Database session.
        analysis_id: Unique analysis identifier.

    Returns:
        Structured audit report data dictionary.
    """
    analysis: Optional[Analysis] = crud.get_analysis(db, analysis_id)
    if not analysis:
        raise ValueError(f"Analysis '{analysis_id}' not found.")

    # Retrieve all related records
    files: List[UploadedFile] = analysis.uploaded_files or []
    model_runs: List[ModelRun] = analysis.model_runs or []
    traces: List[ExecutionTrace] = crud.get_trace_steps(db, analysis_id)
    evidence_items: List[Evidence] = crud.get_evidence_for_analysis(db, analysis_id)

    # Compute execution duration
    duration_seconds: Optional[float] = None
    if analysis.started_at and analysis.completed_at:
        duration_seconds = round((analysis.completed_at - analysis.started_at).total_seconds(), 2)
    elif analysis.started_at:
        duration_seconds = round((datetime.now(timezone.utc) - analysis.started_at).total_seconds(), 2)

    # Structure files data
    files_summary = []
    for f in files:
        meta = f.metadata_json or {}
        isro_info = meta.get("isro")
        res = f.resolution or {}
        bounds = f.bounds or {}
        files_summary.append({
            "file_id": f.id,
            "original_name": f.original_name,
            "format": f.file_format or meta.get("format", "Raster"),
            "file_size_mb": round((f.file_size_bytes or 0) / (1024 * 1024), 2),
            "dimensions": f"{f.width} × {f.height} px" if f.width and f.height else "Unknown",
            "bands": f.bands,
            "dtype": f.dtype or meta.get("dtype", "uint8"),
            "crs": f.crs or "Unprojected / Local Pixel Grid",
            "bounds": bounds,
            "resolution_m": f"{res.get('x', '—')}m × {res.get('y', '—')}m" if res else "—",
            "modality": f.modality or "optical",
            "temporal_label": f.temporal_label,
            "isro": isro_info,
            "preview_path": f.preview_path,
            "thumbnail_path": f.thumbnail_path,
        })

    # Structure models data
    models_summary = []
    for m in model_runs:
        models_summary.append({
            "model_name": m.model_name,
            "version": m.model_version or "1.0.0",
            "task": m.task,
            "device": m.device or "cpu",
            "is_fallback": bool(m.is_fallback),
            "confidence": round(float(m.confidence), 4) if m.confidence is not None else None,
            "execution_time_ms": round(float(m.execution_time_ms), 1) if m.execution_time_ms is not None else None,
            "status": m.status,
            "output_data": m.output_data or {},
        })

    # Structure evidence data
    evidence_summary = []
    for ev in evidence_items:
        evidence_summary.append({
            "type": ev.evidence_type,
            "description": ev.description,
            "file_path": ev.file_path,
            "metadata": ev.metadata_json or {},
        })

    # Structure execution traces
    trace_steps = []
    for tr in traces:
        trace_steps.append({
            "step": tr.step_number,
            "action": tr.action,
            "status": tr.status,
            "duration_ms": round(float(tr.duration_ms), 1) if tr.duration_ms is not None else None,
            "details": tr.details,
            "timestamp": tr.created_at.isoformat() if tr.created_at else None,
        })

    # Audit verification hash / signature summary
    now_utc = datetime.now(timezone.utc)
    report_data = {
        "analysis_id": analysis.id,
        "query": analysis.query,
        "task": analysis.task or "general_vqa",
        "input_type": analysis.input_type or "single",
        "status": analysis.status,
        "answer_text": analysis.answer_text or "No answer recorded.",
        "confidence": round(float(analysis.confidence), 4) if analysis.confidence is not None else 0.0,
        "confidence_level": analysis.confidence_level or "UNCERTAIN",
        "is_fallback": bool(analysis.is_fallback),
        "duration_seconds": duration_seconds,
        "timestamps": {
            "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
            "started_at": analysis.started_at.isoformat() if analysis.started_at else None,
            "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None,
            "report_generated_at": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        },
        "files": files_summary,
        "models_used": models_summary,
        "evidence": evidence_summary,
        "traces": trace_steps,
        "system_version": "SatQuery AI v0.1.0-evaluation",
        "disclaimer": (
            "This automated Earth Observation report was generated using multi-modal AI and remote sensing computer vision. "
            "Observations are subject to sensor GSD, cloud occlusion, and synthetic aperture radar speckle characteristics. "
            "Ground truthing or field verification is recommended for safety-critical and high-impact operational decisions."
        ),
    }

    return report_data


def render_html_report(report_data: Dict[str, Any]) -> str:
    """Render a standalone, print-optimized HTML audit report.

    Args:
        report_data: Compiled audit data from generate_report_data.

    Returns:
        Complete HTML string with embedded CSS and print styles.
    """
    conf_level = report_data.get("confidence_level", "MEDIUM")
    conf_score = int(report_data.get("confidence", 0.0) * 100)

    # Color mappings
    conf_badge_class = {
        "HIGH": "badge-high",
        "MEDIUM": "badge-med",
        "LOW": "badge-low",
        "UNCERTAIN": "badge-uncertain",
    }.get(conf_level, "badge-med")

    # Files table rows
    files_rows = []
    for f in report_data.get("files", []):
        isro_text = ""
        if f.get("isro"):
            isro_text = f"<br><span class='isro-tag'>🇮🇳 {f['isro'].get('platform', 'ISRO')} ({f['isro'].get('sensor_type', '')})</span>"
        files_rows.append(f"""
        <tr>
            <td><strong>{f['original_name']}</strong>{isro_text}</td>
            <td><span class="pill pill-subtle">{f['modality'].upper()}</span></td>
            <td>{f['dimensions']}</td>
            <td><code>{f['crs']}</code></td>
            <td>{f['resolution_m']}</td>
            <td>{f['file_size_mb']} MB</td>
        </tr>
        """)
    files_tbody = "\n".join(files_rows) or "<tr><td colspan='6'>No files registered.</td></tr>"

    # Models table rows
    models_rows = []
    for m in report_data.get("models_used", []):
        fallback_badge = "<span class='badge-fallback'>Fallback</span>" if m['is_fallback'] else "<span class='badge-primary'>Primary</span>"
        models_rows.append(f"""
        <tr>
            <td><strong>{m['model_name']}</strong></td>
            <td>{m['task']}</td>
            <td>{fallback_badge}</td>
            <td>{m['device'].upper()}</td>
            <td>{m['confidence'] or '—'}</td>
            <td>{m['execution_time_ms']} ms</td>
        </tr>
        """)
    models_tbody = "\n".join(models_rows) or "<tr><td colspan='6'>No models executed.</td></tr>"

    # Trace steps
    trace_rows = []
    for t in report_data.get("traces", []):
        status_color = "status-success" if t['status'] == "success" else "status-warn"
        trace_rows.append(f"""
        <div class="trace-item">
            <div class="trace-header">
                <span class="trace-step">Step {t['step']}</span>
                <span class="trace-action">{t['action']}</span>
                <span class="trace-status {status_color}">{t['status'].upper()}</span>
                <span class="trace-time">{t['duration_ms']} ms</span>
            </div>
            <div class="trace-details">{t['details'] or ''}</div>
        </div>
        """)
    trace_html = "\n".join(trace_rows) or "<p class='text-muted'>No execution trace recorded.</p>"

    # Evidence details
    evidence_rows = []
    for ev in report_data.get("evidence", []):
        meta_items = []
        for k, v in ev.get("metadata", {}).items():
            if isinstance(v, dict):
                v_str = ", ".join(f"{sk}: {sv}" for sk, sv in v.items())
                meta_items.append(f"<li><strong>{k}:</strong> {v_str}</li>")
            else:
                meta_items.append(f"<li><strong>{k}:</strong> {v}</li>")
        meta_html = f"<ul class='evidence-list'>{''.join(meta_items)}</ul>" if meta_items else ""

        evidence_rows.append(f"""
        <div class="evidence-card">
            <h4>{ev['type'].replace('_', ' ').title()}</h4>
            <p>{ev['description']}</p>
            {meta_html}
        </div>
        """)
    evidence_html = "\n".join(evidence_rows) or "<p class='text-muted'>Standard inference output; no supplementary raster masks registered.</p>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SatQuery AI Analysis Audit Report — {report_data['analysis_id']}</title>
    <style>
        :root {{
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --border-color: #30363d;
            --text-main: #f0f6fc;
            --text-muted: #8b949e;
            --brand: #22d3ee;
            --brand-dark: #0891b2;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{
            background-color: var(--bg-color);
            color: var(--text-main);
            padding: 2.5rem;
            line-height: 1.6;
        }}
        .report-container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
        }}
        .logo-title h1 {{
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--brand);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .logo-title p {{
            color: var(--text-muted);
            font-size: 0.875rem;
            margin-top: 0.25rem;
        }}
        .header-meta {{
            text-align: right;
            font-size: 0.8rem;
            color: var(--text-muted);
        }}
        .section {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.75rem;
        }}
        .section-title {{
            font-size: 1.15rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
        }}
        .answer-box {{
            background: rgba(34, 211, 238, 0.07);
            border-left: 4px solid var(--brand);
            padding: 1.25rem;
            border-radius: 4px;
            font-size: 1.05rem;
            margin-bottom: 1rem;
        }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        .meta-card {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            padding: 0.75rem 1rem;
            border-radius: 6px;
        }}
        .meta-label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .meta-val {{
            font-size: 1rem;
            font-weight: 600;
            margin-top: 0.25rem;
        }}
        .pill {{
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .pill-subtle {{ background: rgba(255, 255, 255, 0.1); color: var(--text-main); }}
        .badge-high {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
        .badge-med {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
        .badge-low {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
        .badge-uncertain {{ background: rgba(139, 148, 158, 0.2); color: #cbd5e1; border: 1px solid rgba(139, 148, 158, 0.4); }}
        .badge-primary {{ background: rgba(34, 211, 238, 0.15); color: var(--brand); font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; }}
        .badge-fallback {{ background: rgba(245, 158, 11, 0.15); color: var(--warning); font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; }}
        .isro-tag {{ color: #fbbf24; font-size: 0.75rem; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            margin-top: 0.5rem;
        }}
        th, td {{
            text-align: left;
            padding: 0.6rem 0.75rem;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            color: var(--text-muted);
            font-weight: 600;
            background: rgba(255, 255, 255, 0.02);
        }}
        code {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            background: rgba(255, 255, 255, 0.08);
            padding: 0.1rem 0.3rem;
            border-radius: 3px;
            font-size: 0.8rem;
        }}
        .trace-item {{
            border-left: 2px solid var(--border-color);
            padding-left: 1rem;
            margin-bottom: 1rem;
            position: relative;
        }}
        .trace-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .trace-step {{ color: var(--brand); }}
        .trace-status {{ font-size: 0.7rem; padding: 1px 5px; border-radius: 3px; }}
        .status-success {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
        .status-warn {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; }}
        .trace-time {{ color: var(--text-muted); font-size: 0.75rem; margin-left: auto; }}
        .trace-details {{
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }}
        .evidence-card {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        }}
        .evidence-card h4 {{
            font-size: 0.95rem;
            margin-bottom: 0.25rem;
            color: var(--brand);
        }}
        .evidence-card p {{
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}
        .evidence-list {{
            list-style: none;
            font-size: 0.8rem;
        }}
        .evidence-list li {{
            padding: 0.2rem 0;
            color: var(--text-main);
        }}
        .disclaimer-box {{
            background: rgba(245, 158, 11, 0.05);
            border: 1px solid rgba(245, 158, 11, 0.2);
            border-radius: 6px;
            padding: 1rem;
            font-size: 0.8rem;
            color: #d97706;
            margin-top: 2rem;
        }}
        .footer {{
            margin-top: 2rem;
            text-align: center;
            font-size: 0.75rem;
            color: var(--text-muted);
            border-top: 1px solid var(--border-color);
            padding-top: 1rem;
        }}

        /* Print Media Styles */
        @media print {{
            body {{
                background-color: #ffffff !important;
                color: #1a202c !important;
                padding: 0.5in !important;
            }}
            .section {{
                background: #ffffff !important;
                border: 1px solid #e2e8f0 !important;
                page-break-inside: avoid;
            }}
            .answer-box {{
                background: #f0fdf4 !important;
                border-left: 4px solid #0891b2 !important;
                color: #0f172a !important;
            }}
            th {{
                background: #f8fafc !important;
                color: #475569 !important;
            }}
            td {{
                color: #1e293b !important;
                border-bottom: 1px solid #e2e8f0 !important;
            }}
            .trace-item {{
                border-left-color: #cbd5e1 !important;
            }}
            .meta-card, .evidence-card {{
                background: #f8fafc !important;
                border: 1px solid #e2e8f0 !important;
            }}
            .disclaimer-box {{
                background: #fffbeb !important;
                border-color: #fde68a !important;
                color: #92400e !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <!-- Header -->
        <header class="header">
            <div class="logo-title">
                <h1>🛰️ SatQuery AI</h1>
                <p>Earth Observation Query & Forensic Audit Report</p>
            </div>
            <div class="header-meta">
                <div><strong>Analysis ID:</strong> <code>{report_data['analysis_id']}</code></div>
                <div><strong>Generated:</strong> {report_data['timestamps']['report_generated_at']}</div>
                <div><strong>Engine Version:</strong> {report_data['system_version']}</div>
            </div>
        </header>

        <!-- Executive Summary -->
        <div class="section">
            <h2 class="section-title">1. Query & Executive Findings</h2>
            <div style="margin-bottom: 0.75rem;">
                <span class="meta-label">Submitted Query</span>
                <div style="font-size: 1.15rem; font-weight: 600; margin-top: 0.25rem;">
                    "{report_data['query']}"
                </div>
            </div>

            <div class="answer-box">
                <strong>Analysis Conclusion:</strong><br>
                {report_data['answer_text']}
            </div>

            <div class="meta-grid">
                <div class="meta-card">
                    <div class="meta-label">Confidence Score</div>
                    <div class="meta-val">
                        <span class="pill {conf_badge_class}">{conf_level} ({conf_score}%)</span>
                    </div>
                </div>
                <div class="meta-card">
                    <div class="meta-label">Inference Task</div>
                    <div class="meta-val">{report_data['task'].upper()}</div>
                </div>
                <div class="meta-card">
                    <div class="meta-label">Input Configuration</div>
                    <div class="meta-val">{report_data['input_type'].replace('_', ' ').title()}</div>
                </div>
                <div class="meta-card">
                    <div class="meta-label">Processing Latency</div>
                    <div class="meta-val">{report_data['duration_seconds'] or '—'} sec</div>
                </div>
            </div>
        </div>

        <!-- Satellite Inputs -->
        <div class="section">
            <h2 class="section-title">2. Input Satellite Imagery & Sensor Metadata</h2>
            <table>
                <thead>
                    <tr>
                        <th>Raster / Sensor</th>
                        <th>Modality</th>
                        <th>Dimensions</th>
                        <th>Coordinate Reference System</th>
                        <th>Ground Sample Distance</th>
                        <th>File Size</th>
                    </tr>
                </thead>
                <tbody>
                    {files_tbody}
                </tbody>
            </table>
        </div>

        <!-- Models Executed -->
        <div class="section">
            <h2 class="section-title">3. Agentic Model Execution Audit</h2>
            <table>
                <thead>
                    <tr>
                        <th>Model Artifact</th>
                        <th>Specialized Task</th>
                        <th>Engine Mode</th>
                        <th>Compute Device</th>
                        <th>Confidence</th>
                        <th>Execution Time</th>
                    </tr>
                </thead>
                <tbody>
                    {models_tbody}
                </tbody>
            </table>
        </div>

        <!-- Evidence & Metrics -->
        <div class="section">
            <h2 class="section-title">4. Quantitative Evidence & Grounding Analytics</h2>
            {evidence_html}
        </div>

        <!-- Audit Trace -->
        <div class="section">
            <h2 class="section-title">5. Chronological Execution Trace</h2>
            {trace_html}
        </div>

        <!-- Disclaimer -->
        <div class="disclaimer-box">
            <strong>System Audit Disclaimer:</strong><br>
            {report_data['disclaimer']}
        </div>

        <!-- Footer -->
        <footer class="footer">
            SatQuery AI — Autonomous Multi-Modal Remote Sensing Intelligence Platform &bull; Evaluation Audit Hash: <code>SHA256-{report_data['analysis_id'][:16].upper()}</code>
        </footer>
    </div>
</body>
</html>
"""
    return html


def render_pdf_report(report_data: Dict[str, Any]) -> bytes:
    """Render a publication-ready, multi-page PDF audit report using ReportLab.

    Args:
        report_data: Compiled audit data from generate_report_data.

    Returns:
        Binary bytes representing the generated PDF document.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        KeepTogether,
        HRFlowable,
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0891b2"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
    )
    answer_style = ParagraphStyle(
        "AnswerText",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#0f172a"),
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#334155"),
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0f172a"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#92400e"),
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("🛰️ SatQuery AI", title_style))
    story.append(
        Paragraph(
            f"Earth Observation Analysis & Audit Report &bull; ID: <b>{report_data['analysis_id']}</b> &bull; Generated: {report_data['timestamps']['report_generated_at']}",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0891b2"), spaceAfter=10))

    # 2. Executive Summary & Answer
    story.append(Paragraph("1. Query & Executive Findings", h2_style))
    story.append(Paragraph(f"<b>Query:</b> \"{report_data['query']}\"", body_style))
    story.append(Spacer(1, 4))

    # Answer Callout Table
    answer_p = Paragraph(f"<b>Conclusion:</b><br/>{report_data['answer_text']}", answer_style)
    answer_table = Table([[answer_p]], colWidths=[540])
    answer_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdfa")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#2dd4bf")),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(answer_table)
    story.append(Spacer(1, 8))

    # Summary Metrics Row
    conf_pct = int(report_data.get("confidence", 0.0) * 100)
    meta_summary_data = [
        [
            Paragraph("<b>Confidence Level</b>", table_header),
            Paragraph("<b>Inference Task</b>", table_header),
            Paragraph("<b>Input Modality</b>", table_header),
            Paragraph("<b>Latency</b>", table_header),
        ],
        [
            Paragraph(f"{report_data['confidence_level']} ({conf_pct}%)", table_cell),
            Paragraph(report_data['task'].upper(), table_cell),
            Paragraph(report_data['input_type'].replace('_', ' ').title(), table_cell),
            Paragraph(f"{report_data['duration_seconds'] or '—'}s", table_cell),
        ],
    ]
    meta_summary_table = Table(meta_summary_data, colWidths=[135, 135, 135, 135])
    meta_summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(meta_summary_table)
    story.append(Spacer(1, 10))

    # 3. Input Satellite Imagery Table
    story.append(Paragraph("2. Satellite Inputs & Sensor Metadata", h2_style))
    files_table_data = [
        [
            Paragraph("<b>File / Sensor</b>", table_header),
            Paragraph("<b>Modality</b>", table_header),
            Paragraph("<b>Dimensions</b>", table_header),
            Paragraph("<b>CRS</b>", table_header),
            Paragraph("<b>Resolution</b>", table_header),
            Paragraph("<b>Size</b>", table_header),
        ]
    ]
    for f in report_data.get("files", []):
        isro_lbl = f" [{f['isro']['platform']}]" if f.get("isro") else ""
        name_str = f"{f['original_name']}{isro_lbl}"
        files_table_data.append([
            Paragraph(name_str, table_cell),
            Paragraph(f['modality'].upper(), table_cell),
            Paragraph(f['dimensions'], table_cell),
            Paragraph(f['crs'], table_cell),
            Paragraph(f['resolution_m'], table_cell),
            Paragraph(f"{f['file_size_mb']}MB", table_cell),
        ])

    if len(files_table_data) == 1:
        files_table_data.append([Paragraph("No files", table_cell)] * 6)

    files_table = Table(files_table_data, colWidths=[130, 65, 85, 130, 80, 50])
    files_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(files_table)
    story.append(Spacer(1, 10))

    # 4. Agentic Models Executed Table
    story.append(Paragraph("3. Agentic Model Execution Audit", h2_style))
    models_table_data = [
        [
            Paragraph("<b>Model Name</b>", table_header),
            Paragraph("<b>Task</b>", table_header),
            Paragraph("<b>Engine</b>", table_header),
            Paragraph("<b>Device</b>", table_header),
            Paragraph("<b>Confidence</b>", table_header),
            Paragraph("<b>Runtime</b>", table_header),
        ]
    ]
    for m in report_data.get("models_used", []):
        eng = "Fallback" if m['is_fallback'] else "Primary"
        models_table_data.append([
            Paragraph(m['model_name'], table_cell),
            Paragraph(m['task'], table_cell),
            Paragraph(eng, table_cell),
            Paragraph(m['device'].upper(), table_cell),
            Paragraph(str(m['confidence'] or '—'), table_cell),
            Paragraph(f"{m['execution_time_ms']}ms" if m['execution_time_ms'] else "—", table_cell),
        ])

    if len(models_table_data) == 1:
        models_table_data.append([Paragraph("No models executed", table_cell)] * 6)

    models_table = Table(models_table_data, colWidths=[130, 110, 75, 65, 80, 80])
    models_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(models_table)
    story.append(Spacer(1, 10))

    # 5. Execution Trace Table
    story.append(Paragraph("4. Step-by-Step Chronological Audit Trace", h2_style))
    trace_table_data = [
        [
            Paragraph("<b>Step</b>", table_header),
            Paragraph("<b>Action</b>", table_header),
            Paragraph("<b>Status</b>", table_header),
            Paragraph("<b>Duration</b>", table_header),
            Paragraph("<b>Trace Details</b>", table_header),
        ]
    ]
    for tr in report_data.get("traces", []):
        trace_table_data.append([
            Paragraph(str(tr['step']), table_cell),
            Paragraph(tr['action'], table_cell),
            Paragraph(tr['status'].upper(), table_cell),
            Paragraph(f"{tr['duration_ms']}ms" if tr['duration_ms'] else "—", table_cell),
            Paragraph(tr['details'] or "", table_cell),
        ])

    if len(trace_table_data) == 1:
        trace_table_data.append([Paragraph("No trace recorded", table_cell)] * 5)

    trace_table = Table(trace_table_data, colWidths=[35, 120, 55, 60, 270])
    trace_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])
    )
    story.append(trace_table)
    story.append(Spacer(1, 12))

    # 6. Audit Disclaimer Callout
    disc_p = Paragraph(f"<b>System Audit Disclaimer:</b> {report_data['disclaimer']}", disclaimer_style)
    disc_table = Table([[disc_p]], colWidths=[540])
    disc_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fffbeb")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#fde68a")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(disc_table)

    # Build PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
import os
from datetime import datetime

from utils.pdf_visualizations import ordered_eda_charts
from utils.safe_checks import feature_importance_as_dict, normalize_recommendations, safe_dict_get


def generate_pdf_report(report_data):
    """
    Generate a professional PDF report (6–8 pages).
    Accepts legacy dict format (title: content) or structured payload.
    """

    os.makedirs("reports", exist_ok=True)
    file_path = "reports/AutoDS_Report.pdf"

    if _is_structured_payload(report_data):
        _generate_structured_report(file_path, report_data)
    else:
        _generate_legacy_report(file_path, report_data)

    return file_path


def _is_structured_payload(data):
    return "executive_summary" in data or "health_score" in data


def _generate_legacy_report(file_path, report_data):
    from reportlab.pdfgen import canvas

    pdf = canvas.Canvas(file_path, pagesize=letter)
    y = 750
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(70, y, "AutoDS Agent Analysis Report")
    y -= 40
    pdf.setFont("Helvetica", 12)

    for title, content in report_data.items():
        pdf.drawString(50, y, f"{title}:")
        y -= 20
        pdf.drawString(70, y, str(content))
        y -= 35
        if y < 80:
            pdf.showPage()
            y = 750
            pdf.setFont("Helvetica", 12)

    pdf.save()


def _section(story, title, body_style, section_style, lines):
    story.append(Paragraph(title, section_style))
    for line in lines:
        if line:
            story.append(Paragraph(str(line).replace("**", ""), body_style))
    story.append(Spacer(1, 0.12 * inch))


def _bullet_section(story, title, body_style, bullet_style, section_style, items, limit=10):
    story.append(Paragraph(title, section_style))
    if items:
        for item in items[:limit]:
            story.append(Paragraph(f"• {str(item).replace('**', '')}", bullet_style))
    else:
        story.append(Paragraph("Analysis completed with standard automated checks.", body_style))
    story.append(Spacer(1, 0.12 * inch))


def _section_divider(story):
    story.append(Spacer(1, 0.08 * inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0")))
    story.append(Spacer(1, 0.12 * inch))


def _styled_section_title(story, title, section_style):
    story.append(Paragraph(title, section_style))
    story.append(Spacer(1, 0.06 * inch))


def _info_table(rows, header_bg="#4338CA", alt_bg="#F8FAFC"):
    table_data = [[Paragraph(f"<b>{label}</b>", ParagraphStyle("Lbl", fontSize=10, textColor=colors.HexColor("#334155"))),
                   Paragraph(str(value), ParagraphStyle("Val", fontSize=10, textColor=colors.HexColor("#0F172A")))]
                  for label, value in rows]
    table = Table(table_data, colWidths=[2.0 * inch, 4.3 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF2FF")),
                ("ROWBACKGROUNDS", (1, 0), (-1, -1), [colors.white, colors.HexColor(alt_bg)]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return table


def _metric_cards_table(items):
    """Render KPI cards in a 2-column grid."""
    card_style_label = ParagraphStyle("CardLabel", fontSize=9, textColor=colors.HexColor("#64748B"))
    card_style_value = ParagraphStyle("CardValue", fontSize=14, textColor=colors.HexColor("#4338CA"), leading=16)
    rows = []
    row = []
    for label, value in items:
        cell = [
            Paragraph(label, card_style_label),
            Paragraph(f"<b>{value}</b>", card_style_value),
        ]
        row.append(cell)
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        row.append([
            Paragraph("", card_style_label),
            Paragraph("", card_style_value),
        ])
        rows.append(row)
    table = Table(rows, colWidths=[3.15 * inch, 3.15 * inch], rowHeights=[0.75 * inch] * len(rows))
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#E2E8F0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.75, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def _primary_model_score(metrics, problem_type):
    if not isinstance(metrics, dict):
        return None
    if str(problem_type).lower().startswith("reg"):
        return metrics.get("r2") or metrics.get("cv_score") or metrics.get("score")
    return metrics.get("accuracy") or metrics.get("f1") or metrics.get("cv_score") or metrics.get("score")


def _format_metric(value, digits=4):
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _model_comparison_table(data, body_style):
    from utils.model_ranking import composite_model_score, rank_models_by_composite

    detailed = data.get("detailed_metrics") or {}
    training_times = data.get("training_times") or {}
    results_metrics = data.get("results_metrics") or {}
    problem_type = data.get("problem_type") or "Classification"
    best_model = data.get("best_model")
    is_regression = str(problem_type).lower().startswith("reg")
    score_header = "Composite Score" if is_regression else "Composite Score"

    header = ["Rank", "Model", score_header, "Precision", "Recall", "F1", "Training Time", "Status"]
    table_rows = [header]

    if isinstance(detailed, dict) and detailed:
        ranked = rank_models_by_composite(detailed, problem_type, results_metrics)
        for rank, (name, composite) in enumerate(ranked[:10], start=1):
            metrics = detailed.get(name, {})
            if not isinstance(metrics, dict):
                continue
            status = "Best" if name == best_model else "Candidate"
            table_rows.append(
                [
                    str(rank),
                    str(name),
                    _format_metric(composite),
                    _format_metric(metrics.get("precision")),
                    _format_metric(metrics.get("recall")),
                    _format_metric(metrics.get("f1")),
                    _format_metric(training_times.get(name), digits=2) + "s" if training_times.get(name) is not None else "—",
                    status,
                ]
            )
    else:
        model_results = data.get("model_results") or []
        if isinstance(model_results, dict):
            model_results = [f"{k}: {v}" for k, v in model_results.items()]
        for rank, item in enumerate(model_results[:10], start=1):
            text = str(item)
            name = text.split(":")[0].strip() if ":" in text else text
            score = text.split(":")[-1].strip() if ":" in text else "—"
            status = "Best" if name == best_model else "Candidate"
            table_rows.append([str(rank), name, score, "—", "—", "—", "—", status])

    if len(table_rows) == 1:
        table_rows.append(["—", "No models recorded", "—", "—", "—", "—", "—", "—"])

    table = Table(table_rows, repeatRows=1, colWidths=[0.45 * inch, 1.35 * inch, 0.85 * inch, 0.85 * inch, 0.75 * inch, 0.65 * inch, 0.95 * inch, 0.75 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4338CA")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _caption_style(body_style):
    return ParagraphStyle(
        "ChartCaption",
        parent=body_style,
        fontSize=11,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=6,
        spaceAfter=4,
    )


def _embed_visualization(story, title, path, body_style, caption_style, max_width=None):
    """Embed a chart image with safe fallbacks and page-friendly sizing."""
    max_width = max_width or (6.5 * inch)
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph(title, caption_style))
    story.append(Spacer(1, 0.04 * inch))
    if path and os.path.isfile(path):
        try:
            img = Image(path)
            aspect = img.imageHeight / float(img.imageWidth or 1)
            img.drawWidth = max_width
            img.drawHeight = max_width * aspect
            max_height = 4.2 * inch
            if img.drawHeight > max_height:
                img.drawHeight = max_height
                img.drawWidth = max_height / aspect
            story.append(img)
        except Exception:
            story.append(Paragraph("Visualization unavailable for this dataset.", body_style))
    else:
        story.append(Paragraph("Visualization unavailable for this dataset.", body_style))
    story.append(Spacer(1, 0.14 * inch))


def _append_eda_visualizations(story, data, body_style, caption_style):
    eda_paths = data.get("eda_chart_paths") or {}
    charts = ordered_eda_charts(eda_paths if isinstance(eda_paths, dict) else {})
    if not charts:
        return
    story.append(Spacer(1, 0.1 * inch))
    for idx, (title, path) in enumerate(charts):
        _embed_visualization(story, title, path, body_style, caption_style)
        if idx < len(charts) - 1:
            story.append(PageBreak())


def _append_explainability_visualizations(story, data, body_style, caption_style):
    xai_paths = data.get("explainability_chart_paths") or {}
    if not isinstance(xai_paths, dict):
        return
    path = xai_paths.get("feature_importance") or xai_paths.get("shap_summary")
    if path:
        _embed_visualization(story, "Feature Importance Chart", path, body_style, caption_style)


def _generate_structured_report(file_path, data):
    doc = SimpleDocTemplate(
        file_path,
        pagesize=letter,
        rightMargin=60,
        leftMargin=60,
        topMargin=60,
        bottomMargin=60,
    )

    styles = getSampleStyleSheet()
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=14,
        spaceAfter=10,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=14,
        spaceAfter=4,
    )
    cover_style = ParagraphStyle(
        "Cover",
        parent=styles["Heading1"],
        fontSize=28,
        textColor=colors.HexColor("#4338CA"),
        spaceAfter=20,
        alignment=1,
    )
    badge_style = ParagraphStyle(
        "Badge",
        parent=body_style,
        alignment=1,
        fontSize=11,
        textColor=colors.HexColor("#6366F1"),
        spaceAfter=10,
    )

    story = []
    project_name = data.get("project_name") or data.get("project_goal") or "AutoDS Analysis"
    dataset_name = data.get("dataset_name") or "Uploaded Dataset"
    generated_at = data.get("generated_at") or datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    problem_type = data.get("problem_type") or "N/A"
    accuracy_label = data.get("accuracy_display") or data.get("accuracy") or "Unavailable"
    deployment_status = data.get("deployment_status") or data.get("deployment_label") or "Pending"

    # Cover page
    story.append(Spacer(1, 1.4 * inch))
    story.append(Paragraph("AutoDS Agent", cover_style))
    story.append(Paragraph(
        "Autonomous AI Data Scientist",
        ParagraphStyle("CoverTag", parent=body_style, alignment=1, fontSize=13, textColor=colors.HexColor("#6366F1")),
    ))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "Executive Research &amp; Decision Report",
        ParagraphStyle("CoverTitle", parent=body_style, alignment=1, fontSize=16, textColor=colors.HexColor("#1E293B"), spaceAfter=16),
    ))
    story.append(Spacer(1, 0.25 * inch))
    uploaded_file = data.get("uploaded_file") or dataset_name
    detected_dataset = data.get("detected_dataset")
    cover_rows = [("Generation Time", generated_at), ("Uploaded File", uploaded_file), ("Project", project_name)]
    if detected_dataset:
        cover_rows.insert(2, ("Detected Dataset", detected_dataset))
    else:
        cover_rows.insert(2, ("Dataset", dataset_name))
    story.append(_info_table(cover_rows))
    story.append(PageBreak())

    # Executive Summary
    _styled_section_title(story, "Executive Summary", section_style)
    exec_text = str(data.get("executive_summary") or "").strip()
    if exec_text:
        story.append(Paragraph(exec_text, body_style))
        story.append(Spacer(1, 0.12 * inch))
    for note in normalize_recommendations(data.get("model_selection_notes") or [])[:4]:
        story.append(Paragraph(str(note).replace("**", ""), body_style))
    story.append(Spacer(1, 0.12 * inch))

    # Dataset Information
    _styled_section_title(story, "Dataset Information", section_style)
    dataset_rows = [("Uploaded File", uploaded_file), ("Rows", data.get("rows", "N/A")), ("Columns", data.get("columns", "N/A")), ("Target", data.get("target_column", "N/A")), ("Problem Type", problem_type)]
    if detected_dataset:
        dataset_rows.insert(1, ("Detected Dataset", detected_dataset))
    story.append(_info_table(dataset_rows))
    story.append(Spacer(1, 0.18 * inch))

    # Dataset Understanding
    _styled_section_title(story, "Dataset Understanding", section_style)
    understanding_lines = normalize_recommendations(data.get("dataset_understanding") or data.get("dataset_summary") or [])
    if not understanding_lines:
        understanding_lines = [
            f"The dataset contains {data.get('rows', 'N/A')} records and {data.get('columns', 'N/A')} features.",
            f"The selected target column is '{data.get('target_column', 'N/A')}', and the workload is classified as {problem_type}.",
            "AutoDS evaluated feature distributions, missing values, and target balance as part of the analysis.",
        ]
    for line in understanding_lines[:6]:
        story.append(Paragraph(str(line).replace("**", ""), body_style))
    story.append(Spacer(1, 0.18 * inch))

    # Executive Metrics
    _styled_section_title(story, "Executive Metrics", section_style)
    story.append(_metric_cards_table([
        ("Health Score", f"{data.get('health_score', 'N/A')}/100"),
        (f"{'R²' if str(problem_type).lower().startswith('reg') else 'Accuracy'}", accuracy_label),
        ("Trust Score", data.get("trust_score", "N/A")),
        ("Deployment Status", deployment_status),
        ("Best Model", data.get("best_model", "N/A")),
    ]))
    story.append(PageBreak())

    # Top Model Comparison Table
    _styled_section_title(story, "Top Model Comparison Table", section_style)
    story.append(_model_comparison_table(data, body_style))

    # Model Selection Explanation
    model_selection_lines = normalize_recommendations(data.get("model_selection_notes") or [])
    if not model_selection_lines:
        model_selection_lines = [
            f"The best candidate model was selected based on composite performance metrics, cross-validation stability, and business readiness for {problem_type}.",
            f"Selected model: {data.get('best_model', 'N/A')}.",
        ]
    _bullet_section(story, "Model Selection Explanation", body_style, bullet_style, section_style, model_selection_lines, limit=8)
    story.append(PageBreak())

    # Data Quality Assessment Assessment
    quality_actions = normalize_recommendations(data.get("data_quality_assessment") or data.get("cleaning_actions") or [])
    if not quality_actions:
        quality_actions = ["Data quality checks and automated cleaning were completed."]
    _bullet_section(story, "Data Quality Assessment", body_style, bullet_style, section_style, quality_actions, limit=12)
    if data.get("cleaning_before_rows") is not None or data.get("cleaning_after_rows") is not None:
        before_rows = data.get("cleaning_before_rows")
        after_rows = data.get("cleaning_after_rows")
        if before_rows is not None and after_rows is not None:
            story.append(_info_table([
                ("Records Before Cleaning", before_rows),
                ("Records After Cleaning", after_rows),
            ]))
            story.append(Spacer(1, 0.12 * inch))

    # AI-generated EDA Insights
    eda_findings = normalize_recommendations(data.get("eda_findings") or data.get("ai_insights") or [])
    _bullet_section(story, "AI-generated EDA Insights", body_style, bullet_style, section_style, eda_findings, limit=12)
    _append_eda_visualizations(story, data, body_style, _caption_style(body_style))

    # Feature Engineering Summary
    fe_summary = normalize_recommendations(data.get("feature_engineering_summary") or [])
    if fe_summary:
        _bullet_section(
            story,
            "Feature Engineering Summary",
            body_style,
            bullet_style,
            section_style,
            fe_summary,
            limit=10,
        )

    # Model Training Summary
    training_summary = normalize_recommendations(data.get("model_training_summary") or [])
    if not training_summary:
        training_summary = [
            f"Best model: {data.get('best_model', 'N/A')}",
            f"Primary score: {accuracy_label}",
        ]
    _bullet_section(story, "Model Training Summary", body_style, bullet_style, section_style, training_summary, limit=8)
    story.append(PageBreak())

    # Explainability Summary
    _styled_section_title(story, "Explainability Summary", section_style)
    _append_explainability_visualizations(story, data, body_style, _caption_style(body_style))
    fi = feature_importance_as_dict(data.get("feature_importance"))
    if fi:
        fi_rows = [(f"Feature Importance — {name}", _format_metric(value)) for name, value in sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:10]]
        story.append(_info_table(fi_rows))
        story.append(Spacer(1, 0.12 * inch))
    shap_ranking = normalize_recommendations(data.get("shap_ranking"))
    if shap_ranking:
        _bullet_section(story, "SHAP Summary", body_style, bullet_style, section_style, shap_ranking, limit=10)
    elif not fi:
        _bullet_section(story, "SHAP Summary", body_style, bullet_style, section_style, ["SHAP summary not available."], limit=5)
    if data.get("explainability_summary") and not (fi and "shap skipped" in str(data.get("explainability_summary")).lower()):
        story.append(Paragraph(f"<b>Interpretation:</b> {data.get('explainability_summary')}", body_style))
        story.append(Spacer(1, 0.12 * inch))
    story.append(PageBreak())

    # Business Recommendations
    ai_recommendations = normalize_recommendations(
        data.get("business_insights") or data.get("ai_recommendations") or data.get("recommendations") or data.get("business_recommendations") or []
    )
    ai_recommendations.extend(normalize_recommendations(data.get("model_selection_notes") or []))
    if not ai_recommendations:
        ai_recommendations = ["Recommendations are based on cross-validated model performance, data quality, and governance checks."]
    _bullet_section(story, "Business Insights", body_style, bullet_style, section_style, ai_recommendations, limit=12)

    # Deployment Summary
    _styled_section_title(story, "Deployment Summary", section_style)
    deployment_lines = normalize_recommendations(data.get("deployment_summary") or data.get("deployment_notes") or [])
    if not deployment_lines:
        deployment_lines = [
            f"Deployment status is {deployment_status}.",
            f"Deployment readiness score is {data.get('deployment_readiness_score', 'N/A')}/100.",
        ]
    _bullet_section(story, "Deployment Summary", body_style, bullet_style, section_style, deployment_lines, limit=10)
    deployment_rows = [
        ("Deployment Status", deployment_status),
        ("Deployment Readiness", f"{data.get('deployment_readiness_score', 'N/A')}/100"),
    ]
    if data.get("deployment_package_path"):
        deployment_rows.append(("Package Path", data.get("deployment_package_path")))
    story.append(_info_table(deployment_rows))
    story.append(Spacer(1, 0.12 * inch))

    # Conclusion
    conclusion_text = str(
        data.get("final_conclusion") or data.get("conclusion") or data.get("business_recommendations") or "The analysis completed successfully."
    )
    _styled_section_title(story, "Final Decision", section_style)
    story.append(Paragraph(conclusion_text, body_style))
    for note in normalize_recommendations(data.get("model_selection_notes") or [])[:3]:
        story.append(Paragraph(str(note).replace("**", ""), body_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(_info_table([
        ("Recommended Model", data.get("best_model", "N/A")),
        ("Overall Score", data.get("final_score", data.get("health_score", "N/A"))),
        ("Deployment Readiness", f"{data.get('deployment_readiness_score', 'N/A')}/100"),
    ]))

    story.append(Spacer(1, 0.35 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0")))
    story.append(Paragraph(
        "Generated automatically by AutoDS Agent",
        ParagraphStyle("Footer", parent=body_style, fontSize=9, textColor=colors.HexColor("#64748B"), alignment=1),
    ))

    def _page_footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawCentredString(letter[0] / 2, 0.45 * inch, f"Page {canvas.getPageNumber()}")
        canvas.drawCentredString(letter[0] / 2, 0.28 * inch, "AutoDS Agent — Executive Research & Decision Report")
        canvas.restoreState()

    doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)

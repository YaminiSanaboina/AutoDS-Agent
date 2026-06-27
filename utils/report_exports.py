"""Generate Excel exports from pipeline session data."""

from __future__ import annotations

import io
from typing import Any, Dict, List

import pandas as pd

from utils.safe_checks import coalesce_dict, coalesce_list, normalize_recommendations, safe_dict_get, format_accuracy_display, display_kpi_value


def _leaderboard_rows(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = coalesce_list(ctx.get("leaderboard"))
    if rows:
        return [r for r in rows if isinstance(r, dict)]
    metrics = coalesce_dict(ctx.get("metrics"))
    times = coalesce_dict(ctx.get("training_times"))
    out = []
    for rank, (name, score) in enumerate(sorted(metrics.items(), key=lambda x: x[1], reverse=True), start=1):
        out.append({"Rank": rank, "Model": name, "Score": score, "Training Time": times.get(name, "—")})
    return out


def build_excel_bytes(ctx: Dict[str, Any]) -> bytes:
    """Build multi-sheet Excel workbook from report context."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary = {
            "Field": [
                "Uploaded File",
                "Detected Dataset",
                "Rows",
                "Columns",
                "Target",
                "Problem Type",
                "Best Model",
                "Best Score",
                "Health Score",
                "Trust Score",
                "Deployment Status",
            ],
            "Value": [
                display_kpi_value(ctx.get("uploaded_file") or ctx.get("dataset_name")),
                display_kpi_value(ctx.get("detected_dataset")),
                ctx.get("rows", "Unavailable"),
                ctx.get("columns", "Unavailable"),
                display_kpi_value(ctx.get("target_column")),
                display_kpi_value(ctx.get("problem_type")),
                display_kpi_value(ctx.get("best_model")),
                ctx.get("accuracy_display") or format_accuracy_display(ctx.get("best_score"), ctx.get("problem_type", "Classification")),
                ctx.get("health_score", "Unavailable"),
                f"{float(ctx['trust_score']):.0f}/100" if ctx.get("trust_score") is not None and float(ctx["trust_score"]) > 0 else "Unavailable",
                display_kpi_value(ctx.get("deployment_status")),
            ],
        }
        pd.DataFrame(summary).to_excel(writer, sheet_name="Summary", index=False)

        leaderboard = _leaderboard_rows(ctx)
        if leaderboard:
            pd.DataFrame(leaderboard).to_excel(writer, sheet_name="Model Leaderboard", index=False)

        explanation = coalesce_dict(ctx.get("model_selection_explanation"))
        if explanation:
            selection_rows = []
            for reason in coalesce_list(explanation.get("why_chosen")):
                selection_rows.append({"Selection Rationale": str(reason)})
            for alt in coalesce_list(explanation.get("alternatives")):
                if isinstance(alt, dict):
                    model_name = alt.get("model", "Alternative")
                    reasons = coalesce_list(alt.get("reasons"))
                    selection_rows.append({"Selection Rationale": f"Why {model_name} was not chosen:"})
                    for reason in reasons:
                        selection_rows.append({"Selection Rationale": str(reason)})
            if selection_rows:
                pd.DataFrame(selection_rows).to_excel(writer, sheet_name="Selection Rationale", index=False)

        best_model_metrics = coalesce_dict(coalesce_dict(ctx.get("detailed_metrics")).get(ctx.get("best_model")))
        if best_model_metrics:
            pd.DataFrame({"Metric": list(best_model_metrics.keys()), "Value": list(best_model_metrics.values())}).to_excel(
                writer, sheet_name="Best Model Metrics", index=False
            )

        eda_insights = normalize_recommendations(ctx.get("eda_insights"))
        if eda_insights:
            pd.DataFrame({"EDA Insight": eda_insights}).to_excel(writer, sheet_name="EDA", index=False)

        fi = ctx.get("feature_importance_rows") or []
        if fi:
            pd.DataFrame(fi).to_excel(writer, sheet_name="Feature Importance", index=False)

        recs = normalize_recommendations(ctx.get("recommendations"))
        if recs:
            pd.DataFrame({"Recommendation": recs}).to_excel(writer, sheet_name="Recommendations", index=False)

        trust = normalize_recommendations(ctx.get("trust_concerns"))
        if trust:
            pd.DataFrame({"Trust Note": trust}).to_excel(writer, sheet_name="Trust Analysis", index=False)

    buffer.seek(0)
    return buffer.getvalue()



def build_export_context_from_report_ctx(report_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize interactive report context into export payload."""
    chief = coalesce_dict(report_ctx.get("chief"))
    fi_records = report_ctx.get("feature_importance_records") or []
    problem_type = report_ctx.get("problem_type") or "Classification"
    accuracy_display = report_ctx.get("accuracy_display") or format_accuracy_display(
        report_ctx.get("best_score"), problem_type
    )
    trust_score = report_ctx.get("trust_score")
    if trust_score is None:
        trust_score = chief.get("trust_score")
    return {
        "dataset_name": report_ctx.get("uploaded_file") or report_ctx.get("dataset_name") or chief.get("dataset_name"),
        "uploaded_file": report_ctx.get("uploaded_file") or report_ctx.get("dataset_name"),
        "detected_dataset": report_ctx.get("detected_dataset"),
        "rows": report_ctx.get("rows"),
        "columns": report_ctx.get("columns"),
        "target_column": report_ctx.get("target_column"),
        "problem_type": problem_type,
        "best_model": report_ctx.get("best_model") or chief.get("best_model"),
        "best_score": report_ctx.get("best_score"),
        "accuracy_display": accuracy_display,
        "health_score": report_ctx.get("health_score"),
        "trust_score": trust_score,
        "deployment_status": report_ctx.get("deployment_status") or chief.get("deployment_label"),
        "metrics": report_ctx.get("metrics") or {},
        "training_times": report_ctx.get("training_times") or {},
        "leaderboard": report_ctx.get("leaderboard") or [],
        "eda_insights": report_ctx.get("eda_insights") or [],
        "recommendations": report_ctx.get("recommendations") or chief.get("recommendations") or [],
        "trust_concerns": report_ctx.get("trust_concerns") or [],
        "feature_importance_rows": fi_records,
        "model_selection_explanation": report_ctx.get("model_selection_explanation") or {},
        "best_model_metrics": coalesce_dict(coalesce_dict(report_ctx.get("detailed_metrics")).get(report_ctx.get("best_model"))),
    }

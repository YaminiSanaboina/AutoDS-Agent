"""User-friendly display names for pipeline stages (display only — backend keys unchanged)."""

from __future__ import annotations

from typing import Dict

STAGE_DISPLAY_NAMES: Dict[str, str] = {
    "dataset_upload": "Dataset Upload",
    "dataset_intelligence": "Data Profiling",
    "data_cleaning": "Data Cleaning & Preprocessing",
    "eda": "Exploratory Data Analysis (EDA)",
    "feature_engineering": "Feature Engineering",
    "feature_selection": "Feature Selection",
    "automl": "Model Training",
    "model_comparison": "Model Comparison",
    "best_model_selection": "Best Model Selection",
    "model_evaluation": "Model Evaluation",
    "explainability": "Explainability (Feature Importance / SHAP)",
    "ai_ethics_trust": "Trust Score Assessment",
    "self_improvement": "Model Evaluation",
    "deployment_readiness": "Deployment Readiness Assessment",
    "monitoring": "Production Monitoring",
    "ai_decision": "Final AI Decision",
    "prediction": "Prediction Engine",
    "pdf_report": "Executive Report Generation",
    "hyperparameter_optimization": "Feature Selection",
    "initializing": "Initializing",
}


def format_stage_display(stage_key: str | None) -> str:
    """Return a friendly stage label for UI display."""
    if not stage_key or stage_key in ("—", "-", "None"):
        return "—"
    key = str(stage_key).strip()
    if key in STAGE_DISPLAY_NAMES:
        return STAGE_DISPLAY_NAMES[key]
    normalized = key.lower().replace(" ", "_")
    if normalized in STAGE_DISPLAY_NAMES:
        return STAGE_DISPLAY_NAMES[normalized]
    return key.replace("_", " ").title()

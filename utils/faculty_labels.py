"""Faculty-friendly labels and short explanations for the enterprise UI."""

from __future__ import annotations

NAV_ITEMS = [
    ("home", "Home"),
    ("ai_assistant", "Assistant"),
    ("reports", "Reports"),
]

FACULTY_LABELS = {
    "ai_confidence": "Model Reliability",
    "deployment_readiness": "Production Readiness",
    "explainability": "Feature Impact Analysis",
    "automl_studio": "Model Training Center",
    "ethics": "Trust & Fairness Analysis",
    "eda_lab": "Data Analysis Lab",
    "dataset_hub": "Dataset",
    "reports": "Report Center",
    "ai_assistant": "AI Assistant",
    "home": "Home",
    "dashboard": "Home",
}

FACULTY_HELP = {
    "model_reliability": "This indicates how trustworthy the model predictions are based on validation performance.",
    "production_readiness": "This indicates whether the model is suitable for real-world deployment.",
    "dataset_health": "A composite score based on missing values, duplicates, feature diversity, and row completeness.",
    "feature_impact": "Shows which variables most strongly increase or decrease the model prediction.",
    "trust_fairness": "Evaluates bias risk, fairness, privacy, and compliance before deployment.",
    "model_training": "Compares multiple algorithms and selects the best performer for your dataset.",
}


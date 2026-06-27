"""Dataset-specific live recommendations for the enterprise dashboard."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from utils.health_score import compute_health_score, detect_data_issues
from utils.safe_checks import coalesce_dict, safe_dict_get


def generate_live_recommendations(
    df: Optional[pd.DataFrame],
    output: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Build actionable recommendations from real dataset and pipeline metrics."""
    recommendations: List[str] = []
    if df is None or df.empty:
        return ["Upload a dataset from Dataset Hub to receive personalized recommendations."]

    health = compute_health_score(df)
    for item in health.get("recommendations", [])[:4]:
        recommendations.append(str(item))

    issues = detect_data_issues(df)
    for issue in issues[:5]:
        if issue.get("title") == "No Critical Issues":
            continue
        rec = issue.get("recommendation")
        if rec:
            recommendations.append(f"{issue.get('title')}: {rec}")

    if output:
        model_results = coalesce_dict(safe_dict_get(output, "model_results"))
        best = safe_dict_get(model_results, "best_model")
        metrics = coalesce_dict(safe_dict_get(model_results, "metrics"))
        detailed = coalesce_dict(safe_dict_get(model_results, "detailed_metrics"))
        if best and best in metrics:
            score = float(metrics[best])
            if score <= 1 and score < 0.65:
                recommendations.append(
                    f"Weak model performance ({score * 100:.1f}%): engineer new features or collect more data."
                )
            elif score <= 1 and score < 0.75:
                recommendations.append("Moderate model score: try hyperparameter tuning or ensemble models.")

        deploy = coalesce_dict(safe_dict_get(output, "deployment_readiness"))
        for rec in (deploy.get("recommendations") or [])[:3]:
            recommendations.append(str(rec))

        target = safe_dict_get(output, "training_artifacts", {}).get("target_column") if isinstance(output.get("training_artifacts"), dict) else None
        if target and target in df.columns:
            counts = df[target].value_counts(normalize=True, dropna=True)
            if len(counts) >= 2 and counts.iloc[0] > 0.85:
                recommendations.append(
                    f"Class imbalance detected in '{target}': apply SMOTE or class weighting."
                )

        if detailed:
            spread = max(
                (float(m.get("accuracy", m.get("r2", 0)) or 0) for m in detailed.values() if isinstance(m, dict)),
                default=0,
            ) - min(
                (float(m.get("accuracy", m.get("r2", 0)) or 0) for m in detailed.values() if isinstance(m, dict)),
                default=0,
            )
            if spread < 0.03 and len(detailed) > 2:
                recommendations.append("Models perform similarly: review feature quality and target leakage.")

    seen = set()
    unique: List[str] = []
    for rec in recommendations:
        key = rec.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    return unique[:10] if unique else ["Dataset looks healthy — run analysis to generate model recommendations."]

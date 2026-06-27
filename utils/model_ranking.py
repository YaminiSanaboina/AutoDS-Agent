"""Shared model ranking helpers — one composite formula for leaderboards and reports."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

COMPOSITE_TEST_WEIGHT = 0.7
COMPOSITE_CV_WEIGHT = 0.3
COMPOSITE_FORMULA_LABEL = (
    "Composite production score: 70% hold-out validation metric + 30% cross-validation stability"
)


def _holdout_metric(metrics: Dict[str, Any], problem_type: str) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    if str(problem_type).lower().startswith("reg"):
        for key in ("r2", "score"):
            if metrics.get(key) is not None:
                return float(metrics[key])
    else:
        for key in ("accuracy", "f1", "score"):
            if metrics.get(key) is not None:
                return float(metrics[key])
    return None


def composite_model_score(
    name: str,
    detailed_metrics: Dict[str, Dict[str, Any]],
    problem_type: str,
    results_metrics: Optional[Dict[str, Any]] = None,
) -> float:
    """Match AutoML selection in model_agent.train_selected_models."""
    dm = detailed_metrics.get(name, {}) if isinstance(detailed_metrics, dict) else {}
    cv = dm.get("cv_score")

    if isinstance(results_metrics, dict) and name in results_metrics:
        try:
            primary = float(results_metrics[name])
        except (TypeError, ValueError):
            primary = None
        if primary is not None:
            if cv is not None:
                return COMPOSITE_TEST_WEIGHT * primary + COMPOSITE_CV_WEIGHT * float(cv)
            return primary

    holdout = _holdout_metric(dm, problem_type)
    if holdout is None:
        return 0.0
    if cv is not None:
        return COMPOSITE_TEST_WEIGHT * holdout + COMPOSITE_CV_WEIGHT * float(cv)
    return holdout


def rank_models_by_composite(
    detailed_metrics: Dict[str, Dict[str, Any]],
    problem_type: str,
    results_metrics: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, float]]:
    if not isinstance(detailed_metrics, dict) or not detailed_metrics:
        return []
    ranked = [
        (name, composite_model_score(name, detailed_metrics, problem_type, results_metrics))
        for name in detailed_metrics
        if isinstance(detailed_metrics.get(name), dict) and "error" not in detailed_metrics.get(name, {})
    ]
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def top_ranked_model_name(
    detailed_metrics: Dict[str, Dict[str, Any]],
    problem_type: str,
    results_metrics: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    ranked = rank_models_by_composite(detailed_metrics, problem_type, results_metrics)
    return ranked[0][0] if ranked else None


def build_best_model_consistency_notes(
    best_model: Optional[str],
    detailed_metrics: Dict[str, Dict[str, Any]],
    problem_type: str,
    results_metrics: Optional[Dict[str, Any]] = None,
    explanation: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Explain composite selection, especially when leaderboard #1 differs from best model."""
    if not best_model:
        return []
    ranked = rank_models_by_composite(detailed_metrics, problem_type, results_metrics)
    if not ranked:
        return []

    top_name = ranked[0][0]
    notes: List[str] = [COMPOSITE_FORMULA_LABEL + f" — selected **{best_model}** for production."]

    if top_name and top_name != best_model:
        notes.append(
            f"**{top_name}** leads on hold-out validation alone; **{best_model}** was chosen because the "
            "composite score balances validation performance with cross-validation stability."
        )

    if isinstance(explanation, dict):
        for reason in explanation.get("why_chosen") or []:
            text = str(reason).strip()
            if text and text not in notes:
                notes.append(text)
    return notes[:6]

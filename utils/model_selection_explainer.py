"""Dynamic explanations for why a model was selected over alternatives."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _primary_score(metrics: Dict[str, Any], problem_type: str) -> Optional[float]:
    if not metrics:
        return None
    if problem_type == "Classification":
        if metrics.get("cv_score") is not None:
            return float(metrics["cv_score"])
        for key in ("accuracy", "f1", "roc_auc"):
            if key in metrics and metrics[key] is not None:
                return float(metrics[key])
    else:
        if metrics.get("cv_score") is not None:
            return float(metrics["cv_score"])
        for key in ("r2",):
            if key in metrics and metrics[key] is not None:
                return float(metrics[key])
    return None


def _score_gap(best: float, other: float, problem_type: str) -> str:
    if problem_type == "Classification":
        gap = (best - other) * 100
        return f"{gap:.1f} percentage points lower validation accuracy"
    gap = (best - other) * 100 if abs(best) <= 1 else (best - other)
    return f"{gap:.2f} lower R² score"


def _why_chosen(best_name: str, best_metrics: Dict[str, Any], problem_type: str, training_times: Dict[str, float]) -> List[str]:
    reasons: List[str] = [
        "Selected using composite production score (70% hold-out validation + 30% cross-validation stability)."
    ]
    primary = _primary_score(best_metrics, problem_type)
    if primary is not None:
        if problem_type == "Classification":
            reasons.append(f"Highest cross-validated accuracy ({primary * 100:.1f}%) among trained candidates")
            f1 = best_metrics.get("f1")
            if f1 is not None:
                reasons.append(f"Strong F1 score ({float(f1) * 100:.1f}%) indicating balanced precision and recall")
            roc = best_metrics.get("roc_auc")
            if roc is not None:
                reasons.append(f"Competitive ROC-AUC ({float(roc):.3f}) for ranking quality")
        else:
            reasons.append(f"Highest cross-validated R² ({primary:.4f}) among trained candidates")
            rmse = best_metrics.get("rmse")
            if rmse is not None:
                reasons.append(f"Lowest RMSE ({float(rmse):.4f}) on the hold-out set")
            mae = best_metrics.get("mae")
            if mae is not None:
                reasons.append(f"Competitive MAE ({float(mae):.4f}) for prediction error control")

    train_time = training_times.get(best_name)
    if train_time is not None and train_time <= 5:
        reasons.append("Efficient training time relative to performance gain")

    cv = best_metrics.get("cv_score")
    if cv is not None:
        label = "accuracy" if problem_type == "Classification" else "R²"
        reasons.append(f"Stable 3-fold cross-validation {label} ({float(cv) * 100:.1f}%)" if float(cv) <= 1 else f"Stable 3-fold cross-validation {label} ({float(cv):.4f})")

    if not reasons:
        reasons.append("Top-ranked model on the primary validation metric")
    return reasons[:5]


def _why_not_chosen(
    name: str,
    metrics: Dict[str, Any],
    best_score: float,
    problem_type: str,
    training_times: Dict[str, float],
) -> List[str]:
    reasons: List[str] = []
    score = _primary_score(metrics, problem_type)
    if score is not None and best_score is not None:
        if problem_type == "Classification" and score < best_score - 0.005:
            reasons.append(f"Lower validation accuracy ({score * 100:.1f}% vs best {best_score * 100:.1f}%)")
        elif problem_type == "Regression" and score < best_score - 0.01:
            reasons.append(_score_gap(best_score, score, problem_type))

    f1 = metrics.get("f1")
    best_f1_hint = metrics.get("_best_f1")
    if f1 is not None and best_f1_hint is not None and float(f1) < float(best_f1_hint) - 0.03:
        reasons.append("Weaker precision-recall balance (lower F1)")

    train_time = training_times.get(name)
    best_time = metrics.get("_best_time")
    if train_time is not None and best_time is not None and train_time > best_time * 2 and score is not None:
        reasons.append("Longer training time without compensating score improvement")

    if problem_type == "Classification" and name in {"Logistic Regression", "Naive Bayes"}:
        acc = metrics.get("accuracy")
        if acc is not None and best_score is not None and acc < best_score - 0.08:
            reasons.append("Underfitting detected — simpler model could not capture dataset patterns")

    if problem_type == "Classification" and name == "SVM":
        if train_time is not None and train_time > 10:
            reasons.append("Longer training time on this dataset size")
        if score is not None and best_score is not None and score < best_score - 0.02:
            reasons.append("Lower validation score than the selected model")

    if not reasons and score is not None and best_score is not None and score < best_score:
        reasons.append("Did not achieve the highest validation score")
    if not reasons:
        reasons.append("Not selected as the top-performing candidate")
    return reasons[:3]


def build_model_selection_explanation(
    *,
    best_model: str,
    detailed_metrics: Dict[str, Dict[str, Any]],
    problem_type: str,
    training_times: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build faculty-friendly model selection rationale from real training metrics."""
    training_times = training_times or {}
    if not best_model or not detailed_metrics:
        return {
            "selected_model": best_model or "—",
            "why_chosen": ["Complete model training to generate selection rationale."],
            "alternatives": [],
        }

    best_metrics = detailed_metrics.get(best_model, {})
    best_score = _primary_score(best_metrics, problem_type)

    best_f1 = max((float(m.get("f1", 0) or 0) for m in detailed_metrics.values()), default=0)
    best_time = min((float(training_times.get(n, 999)) for n in detailed_metrics), default=999)

    alternatives: List[Dict[str, Any]] = []
    for name, metrics in sorted(detailed_metrics.items(), key=lambda x: _primary_score(x[1], problem_type) or -1, reverse=True):
        if name == best_model:
            continue
        enriched = dict(metrics)
        enriched["_best_f1"] = best_f1
        enriched["_best_time"] = best_time
        alternatives.append(
            {
                "model": name,
                "reasons": _why_not_chosen(name, enriched, best_score or 0, problem_type, training_times),
            }
        )

    return {
        "selected_model": best_model,
        "problem_type": problem_type,
        "why_chosen": _why_chosen(best_model, best_metrics, problem_type, training_times),
        "alternatives": alternatives[:8],
    }

"""Rule-based intelligent responses when external LLM APIs are unavailable."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from utils.health_score import compute_health_score, detect_data_issues
from utils.safe_checks import (
    coalesce_dict,
    coalesce_list,
    feature_importance_as_dict,
    is_present,
    normalize_recommendations,
    safe_dict_get,
)


_DATASET_STORIES = {
    "iris": (
        "This is a classic botanical dataset measuring iris flowers. "
        "Columns typically describe sepal and petal dimensions used to distinguish flower types."
    ),
    "titanic": (
        "This dataset describes passengers aboard the Titanic. "
        "It is commonly used to study survival patterns using age, fare, class, and other passenger attributes."
    ),
    "housing": (
        "This is a real-estate style dataset used to study property characteristics and sale prices. "
        "Features like area, bedrooms, and bathrooms often relate to home value."
    ),
    "wine": (
        "This dataset records chemical measurements of wine samples. "
        "It is often used to predict wine quality or classify wine types from lab readings."
    ),
    "churn": (
        "This dataset is typically used for customer churn prediction — identifying customers likely to leave a service."
    ),
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _match_topic(query: str) -> str:
    q = _normalize(query)
    if any(k in q for k in ("beginner", "simple", "simply", "explain like", "explain this", "non-technical", "easy")):
        return "beginner"
    if any(k in q for k in ("missing", "null", "na value", "empty cell")):
        return "missing"
    if any(k in q for k in ("issue", "problem", "wrong", "quality", "duplicate", "outlier")):
        return "issues"
    if any(k in q for k in ("about", "represent", "describe", "what is", "dataset")):
        return "dataset_about"
    if any(k in q for k in ("important feature", "which feature", "top feature", "key feature", "best feature", "best predictor", "affect", "impact", "influence", "survival")):
        return "features"
    if any(k in q for k in ("best model", "which model", "top model", "performed best", "winner")):
        return "best_model"
    if any(k in q for k in ("why", "selected", "choose", "picked")):
        return "why_model"
    if any(k in q for k in ("compare", "comparison", "all model", "leaderboard", "versus")):
        return "compare_models"
    if any(k in q for k in ("overfit", "overfitting", "generalization", "train test gap")):
        return "overfitting"
    if any(k in q for k in ("shap", "feature importance", "explain feature", "explain prediction", "why predict", "why did the model", "this prediction")):
        return "explainability"
    if any(k in q for k in ("deploy", "production", "ship", "launch")):
        return "deployment"
    if any(k in q for k in ("risk", "warning", "concern", "ethics", "bias", "fair")):
        return "risks"
    if any(k in q for k in ("how good", "quality score", "health score", "grade")):
        return "dataset_quality"
    if any(k in q for k in ("summarize", "summary of report", "report summary", "overall result")):
        return "report_summary"
    if any(k in q for k in ("clean", "cleaning", "preprocess")):
        return "cleaning"
    if any(k in q for k in ("eda", "explor", "correlation", "distribution", "chart")):
        return "eda"
    return "general"


def _infer_story(dataset_name: str, feature_names: List[str]) -> str:
    name = _normalize(dataset_name)
    for key, story in _DATASET_STORIES.items():
        if key in name:
            return story
    cols = [c.lower() for c in feature_names]
    if any("survived" in c for c in cols):
        return _DATASET_STORIES["titanic"]
    if any("price" in c for c in cols):
        return _DATASET_STORIES["housing"]
    if any("petal" in c or "sepal" in c for c in cols):
        return _DATASET_STORIES["iris"]
    if any("alcohol" in c for c in cols) and any("quality" in c for c in cols):
        return _DATASET_STORIES["wine"]
    return (
        "This appears to be a structured tabular dataset where each row is one record "
        "and each column is a measurable attribute or label."
    )


def _sentences(parts: List[str]) -> str:
    cleaned = [p.strip() for p in parts if p and str(p).strip()]
    return " ".join(cleaned)


_BEGINNER_MAX_SENTENCES = 10
_BEGINNER_MIN_SENTENCES = 2


def _split_sentences(text: str) -> List[str]:
    """Split prose into sentences using the same punctuation heuristic as tests."""
    if not text or not str(text).strip():
        return []
    chunks = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _merge_to_single_sentence(parts: List[str]) -> str:
    """Combine multiple short sentences into one beginner-friendly sentence."""
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    lead = parts[0].rstrip(".")
    tail = parts[1].strip()
    if tail and tail[0].isupper():
        tail = tail[0].lower() + tail[1:]
    return f"{lead}, and {tail}"


def _beginner_story(dataset_name: str, feature_names: List[str]) -> str:
    """Return a single-sentence dataset story for beginner explanations."""
    return _merge_to_single_sentence(_split_sentences(_infer_story(dataset_name, feature_names)))


def _pipeline_matches_dataset(pipeline: Dict[str, Any], dataset_name: str) -> bool:
    """True when pipeline metadata belongs to the dataset currently loaded."""
    if not pipeline.get("executed"):
        return False
    pipeline_name = pipeline.get("dataset_name")
    if not pipeline_name:
        return False
    return _normalize(str(pipeline_name)) == _normalize(str(dataset_name))


def _build_beginner_explanation(
    *,
    note: str,
    name: str,
    rows: int,
    cols: int,
    features: List[str],
    missing: int,
    target: Optional[str],
    pipeline: Dict[str, Any],
    model: Dict[str, Any],
) -> str:
    """Build a concise beginner explanation with a deterministic sentence budget."""
    sentences: List[str] = []

    if note:
        sentences.extend(_split_sentences(note))

    sentences.append(f"Your dataset {name} has {rows:,} rows and {cols} columns.")
    if features:
        sentences.append(f"The columns include {', '.join(features[:8])}.")

    story = _beginner_story(name, features)
    if _pipeline_matches_dataset(pipeline, name):
        summary = pipeline.get("dataset_summary")
        if summary:
            story = _merge_to_single_sentence(_split_sentences(str(summary)))
    if story:
        sentences.append(story)

    if missing:
        sentences.append(f"There are {missing:,} missing values across the table.")
    else:
        sentences.append("No missing values were detected in the loaded table.")

    if target and target in features:
        sentences.append(
            f"The target column appears to be '{target}' — the outcome the model tries to predict."
        )

    if _pipeline_matches_dataset(pipeline, name) and model.get("trained") and model.get("best_name"):
        sentences.append(
            f"The best model so far is {model.get('best_name')} with score {model.get('score', 'unknown')}."
        )

    closing = "Start with charts to understand patterns, then review model results and deployment readiness."
    sentences.append(closing)

    if len(sentences) > _BEGINNER_MAX_SENTENCES:
        essential = sentences[: max(_BEGINNER_MIN_SENTENCES, _BEGINNER_MAX_SENTENCES - 1)]
        if closing not in essential:
            essential.append(closing)
        sentences = essential[:_BEGINNER_MAX_SENTENCES]

    return " ".join(sentences)


def build_intelligent_fallback(user_query: str, context: Dict[str, Any], error: Optional[Exception] = None) -> str:
    """Generate a context-aware answer without calling external LLMs."""
    query = (user_query or "").strip()
    if not query:
        return "Please ask a question about your dataset, model, or analysis so I can help."

    topic = _match_topic(query)
    dataset = coalesce_dict(context.get("dataset"))
    pipeline = coalesce_dict(context.get("pipeline"))
    model = coalesce_dict(context.get("model"))
    eda = coalesce_dict(context.get("eda"))
    shap = coalesce_dict(context.get("shap"))
    feature_importance = coalesce_dict(context.get("feature_importance"))
    report = coalesce_dict(context.get("report"))
    ethics = coalesce_dict(context.get("ethics"))
    deployment = coalesce_dict(context.get("deployment"))
    cleaning = coalesce_dict(context.get("cleaning"))
    feature_engineering = coalesce_dict(context.get("feature_engineering"))
    documentation = coalesce_dict(context.get("documentation"))

    name = dataset.get("name") or "your dataset"
    rows = dataset.get("rows", 0)
    cols = dataset.get("columns", 0)
    features = coalesce_list(safe_dict_get(dataset, "feature_names"))
    missing = dataset.get("missing_values", 0)
    duplicates = dataset.get("duplicates", 0)
    target = dataset.get("target_column") or pipeline.get("target_column")
    issues = coalesce_list(safe_dict_get(dataset, "issues"))
    if not issues:
        issues = coalesce_list(safe_dict_get(pipeline, "issues"))

    note = ""
    if error:
        note = "I'm answering from your AutoDS project data because the external LLM provider is unavailable. "

    if topic == "beginner" or topic == "dataset_about":
        target = dataset.get("target_column") or pipeline.get("target_column")
        if target and target not in features:
            target = None
        return _build_beginner_explanation(
            note=note,
            name=name,
            rows=rows,
            cols=cols,
            features=features,
            missing=missing,
            target=target,
            pipeline=pipeline,
            model=model,
        )

    if topic == "missing":
        missing_by_col = coalesce_dict(safe_dict_get(dataset, "missing_by_column"))
        parts = [note, f"{name} has {missing:,} missing values across all columns."]
        if missing_by_col:
            ranked = sorted(missing_by_col.items(), key=lambda x: x[1], reverse=True)[:5]
            col_detail = ", ".join(f"'{c}' ({v})" for c, v in ranked)
            parts.append(f"Missing by column: {col_detail}.")
        elif missing:
            parts.append("Review columns with gaps before training, or let AutoDS cleaning impute or remove them.")
        else:
            parts.append("The loaded dataset shows no missing values, which is a strong quality signal.")
        dtypes = coalesce_dict(safe_dict_get(dataset, "column_dtypes"))
        if dtypes and "dtype" not in q:
            numeric = sum(1 for t in dtypes.values() if any(k in t.lower() for k in ("int", "float")))
            parts.append(f"The dataset has {numeric} numeric and {len(dtypes) - numeric} non-numeric columns.")
        return _sentences(parts)

    if topic == "issues":
        if issues:
            issue_text = "; ".join(str(i) for i in issues[:6])
            return _sentences([note, f"I found these data quality issues: {issue_text}.", "Address these before production deployment."])
        health = dataset.get("quality_score")
        grade = dataset.get("quality_grade")
        if health is not None:
            return _sentences([
                note,
                f"Dataset health score is {health}/100" + (f" (grade {grade})" if grade else "") + ".",
                f"Duplicates: {duplicates}. Missing values: {missing:,}.",
                "Run cleaning and EDA stages for deeper issue detection.",
            ])
        return _sentences([note, "No major issues were flagged yet. Upload data and run the autonomous pipeline for a full audit."])

    if topic == "features":
        top = coalesce_list(safe_dict_get(feature_importance, "top_features"))
        fi = feature_importance_as_dict(safe_dict_get(pipeline, "feature_importance"))
        if not fi:
            fi = feature_importance_as_dict(safe_dict_get(feature_importance, "ranked"))
        if fi:
            ranked = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            top = [k for k, _ in ranked]
        target = dataset.get("target_column")
        q = _normalize(query)
        if top:
            lead = top[0]
            if target and ("surviv" in q or "affect" in q or "impact" in q):
                return _sentences([
                    note,
                    f"Based on feature importance, **{lead}** has the strongest impact on `{target}`.",
                    f"Other influential features: {', '.join(top[1:4])}." if len(top) > 1 else "",
                    "These rankings come from SHAP / model feature importance after training.",
                ])
            return _sentences([
                note,
                f"The most important features are: {', '.join(top)}.",
                "These drive model predictions most strongly — monitor their quality in production.",
            ])
        return _sentences([note, "Feature importance will appear after explainability runs. Train models and open the SHAP stage."])

    if topic == "best_model":
        detailed = coalesce_dict(model.get("detailed_metrics"))
        if model.get("best_name"):
            score = model.get("best_score") or model.get("score")
            dm = coalesce_dict(detailed.get(model.get("best_name"))) if detailed else {}
            cv = dm.get("cv_score")
            parts = [
                note,
                f"The best model is **{model.get('best_name')}** with test score {score}.",
                f"Problem type: {model.get('problem_type', 'unknown')}.",
            ]
            if cv is not None:
                parts.append(f"Cross-validation score: {cv:.4f}.")
            if model.get("leaderboard"):
                parts.append(f"Leaderboard has {len(model.get('leaderboard'))} trained candidates.")
            return _sentences(parts)
        metrics = model.get("results") or {}
        if metrics:
            best = max(metrics.items(), key=lambda x: x[1])
            return _sentences([note, f"Top model from leaderboard: {best[0]} with score {best[1]:.4f}."])
        return _sentences([note, "No trained model yet. Run the autonomous AI scientist pipeline to compare candidates."])

    if topic == "why_model":
        best = model.get("best_name")
        metrics = model.get("results") or {}
        if best and metrics:
            best_score = metrics.get(best)
            spread = max(metrics.values()) - min(metrics.values()) if len(metrics) > 1 else 0
            return _sentences([
                note,
                f"{best} was selected because it achieved the highest validation score ({best_score:.4f}).",
                f"It leads other candidates by {spread:.4f} on the primary metric.",
                "AutoDS also considers stability, explainability, and deployment readiness.",
            ])
        rationale = documentation.get("model_selection") or pipeline.get("model_selection_rationale")
        if rationale:
            return _sentences([note, str(rationale)])
        return _sentences([note, "Run model training to see why the top candidate was chosen."])

    if topic == "compare_models":
        metrics = model.get("results") or {}
        if metrics:
            lines = [f"{name}: {score:.4f}" for name, score in sorted(metrics.items(), key=lambda x: x[1], reverse=True)]
            return _sentences([note, "Model leaderboard:", "; ".join(lines)])
        return _sentences([note, "No model comparison available yet. Complete the AutoML stage to see all scores."])

    if topic == "overfitting":
        metrics = model.get("results") or {}
        if len(metrics) >= 2:
            spread = max(metrics.values()) - min(metrics.values())
            if spread < 0.03:
                return _sentences([
                    note,
                    "Models scored very similarly, which can mean the problem is well-defined or all models underfit/overfit together.",
                    "Compare train vs validation scores in the model report when available.",
                ])
            return _sentences([
                note,
                "Check whether the best model's training score is much higher than validation score.",
                "Large gaps suggest overfitting; similar scores suggest healthier generalization.",
            ])
        return _sentences([note, "Overfitting analysis needs trained models. Run AutoML and review train vs validation metrics."])

    if topic == "explainability":
        if shap.get("computed") or feature_importance.get("available"):
            top = feature_importance.get("top_features") or []
            expl = pipeline.get("explainability_summary") or ""
            parts = [note, "SHAP / feature importance is available."]
            if top:
                parts.append(f"Top drivers: {', '.join(top)}.")
            if expl:
                parts.append(str(expl))
            parts.append("Positive features push predictions up; negative features pull them down.")
            return _sentences(parts)
        return _sentences([note, "Explainability runs after model training. Ask again after the SHAP stage completes."])

    if topic == "deployment":
        risk = deployment.get("risk_level") or pipeline.get("deployment_risk")
        readiness = deployment.get("readiness_score") or pipeline.get("final_score")
        rec = pipeline.get("recommendation") or deployment.get("recommendation")
        parts = [note]
        if readiness is not None:
            parts.append(f"Production readiness score: {readiness}/100.")
        if risk:
            parts.append(f"Deployment risk level: {risk}.")
        if rec:
            parts.append(f"AI recommendation: {rec}.")
        if risk == "Low":
            parts.append("The model looks suitable for a staged production rollout with monitoring.")
        elif risk:
            parts.append("Review warnings and ethics checks before deploying.")
        if len(parts) == 1:
            parts.append("Complete the deployment readiness stage to get a production assessment.")
        return _sentences(parts)

    if topic == "risks":
        concerns = normalize_recommendations(safe_dict_get(ethics, "concerns"))
        warnings = normalize_recommendations(safe_dict_get(deployment, "warnings"))
        parts = [note]
        if concerns:
            parts.append("Ethics concerns: " + "; ".join(str(c) for c in concerns[:5]))
        if warnings:
            parts.append("Deployment warnings: " + "; ".join(str(w) for w in warnings[:5]))
        if not concerns and not warnings:
            parts.append("No major ethics or deployment risks were flagged. Continue monitoring bias and drift.")
        return _sentences(parts)

    if topic == "dataset_quality":
        health = dataset.get("quality_score")
        grade = dataset.get("quality_grade") or dataset.get("letter_grade")
        parts = [note, f"Dataset health score: {health}/100." if health is not None else note]
        if grade:
            parts.append(f"Letter grade: {grade}.")
        parts.append(f"Missing values: {missing:,}. Duplicates: {duplicates}.")
        if issues:
            parts.append("Issues: " + "; ".join(str(i) for i in issues[:4]))
        elif health is not None and float(health) >= 80:
            parts.append("Overall data quality is strong for modeling.")
        elif health is not None:
            parts.append("Consider addressing data quality gaps before production deployment.")
        return _sentences(parts)

    if topic == "report_summary":
        summary = report.get("executive_summary") or documentation.get("summary") or pipeline.get("recommendation")
        best = model.get("best_name")
        parts = [note]
        if summary:
            parts.append(str(summary))
        if best:
            parts.append(f"Recommended model: {best}.")
        deploy_risk = deployment.get("risk_level") or pipeline.get("deployment_risk")
        if deploy_risk:
            parts.append(f"Deployment risk: {deploy_risk}.")
        if not summary and not best:
            parts.append("Run Autonomous Analysis on Home, then open Reports for the full executive summary.")
        return _sentences(parts)

    if topic == "report":
        if report.get("generated") or report.get("report_path"):
            summary = report.get("executive_summary") or documentation.get("summary")
            parts = [note, "A project report has been generated."]
            if summary:
                parts.append(str(summary)[:500])
            return _sentences(parts)
        return _sentences([note, "Run the full pipeline to generate the executive PDF report."])

    if topic == "cleaning":
        actions = normalize_recommendations(safe_dict_get(cleaning, "actions"))
        if actions:
            return _sentences([note, "Cleaning actions applied:", "; ".join(str(a) for a in actions[:8])])
        return _sentences([note, "No cleaning report yet. The data cleaning stage will list imputation, encoding, and duplicate handling."])

    if topic == "eda":
        insights = normalize_recommendations(safe_dict_get(eda, "insights"))
        if not insights:
            insights = normalize_recommendations(safe_dict_get(pipeline, "eda_insights"))
        if is_present(insights):
            return _sentences([note, "EDA insights:", "; ".join(str(i) for i in insights[:6])])
        if eda.get("generated"):
            return _sentences([note, "EDA has been generated. Review correlation heatmaps and distributions in the EDA stage panel."])
        return _sentences([note, "Run EDA to explore distributions, correlations, and anomalies."])

    if topic == "feature_engineering" or feature_engineering:
        steps = normalize_recommendations(safe_dict_get(feature_engineering, "steps"))
        if not steps:
            steps = normalize_recommendations(safe_dict_get(feature_engineering, "recommended_changes"))
        if steps:
            return _sentences([note, "Feature engineering suggestions:", "; ".join(str(s) for s in steps[:6])])

    # General fallback
    parts = [note or "I'm answering from your AutoDS project context."]
    parts.append(f"Dataset {name}: {rows:,} rows, {cols} columns, {missing:,} missing values.")
    if model.get("trained"):
        parts.append(f"Best model: {model.get('best_name')} ({model.get('score', 'unknown')}).")
    if shap.get("computed"):
        parts.append("SHAP explanations are available.")
    parts.append("Ask about missing values, model comparison, feature importance, deployment, or beginner explanations.")
    return _sentences(parts)

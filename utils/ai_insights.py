import pandas as pd


def generate_model_insights(
    df,
    problem_type,
    best_model_name,
    results,
    shap_importance,
    target_column,
):
    insights = []
    recommendations = []

    if not results:
        return insights, recommendations, 0.0

    scores = list(results.values())
    best_score = max(scores)
    avg_score = sum(scores) / len(scores)
    score_spread = best_score - min(scores)

    metric = "accuracy" if problem_type == "Classification" else "R²"
    # Friendly explanation of best model
    if problem_type == "Classification":
        insights.append(
            f"The best performing model is **{best_model_name}** with {metric} of **{best_score:.2%}**. "
            f"This means the model correctly predicts approximately {int(best_score*100)} out of 100 cases."
        )
    else:
        insights.append(
            f"The best performing model is **{best_model_name}** with R² of **{best_score:.4f}**. "
            f"R² near 1 indicates better fit; here it means the model explains about {best_score*100:.1f}% of the variance."
        )

    if score_spread < 0.03:
        insights.append(
            "Model scores are closely clustered, suggesting the problem may be "
            "well-defined with limited headroom for algorithm selection."
        )
    else:
        insights.append(
            f"**{best_model_name}** outperforms the weakest model by "
            f"{score_spread:.2%}." if problem_type == "Classification"
            else f"**{best_model_name}** outperforms the weakest model by {score_spread:.4f} R²."
        )

    if shap_importance is not None and not shap_importance.empty:
        top_feature = shap_importance.iloc[0]["Feature"]
        top_impact = shap_importance.iloc[0]["Mean |SHAP|"]
        insights.append(
            f"Feature importance: **{top_feature}** is the strongest driver of predictions "
            f"(mean |SHAP| = {top_impact:.4f}). This means changes in this feature strongly affect model output."
        )

        top_three = shap_importance.head(3)["Feature"].tolist()
        insights.append(
            f"The top three influential features are: **{', '.join(top_three)}**. These features should be prioritized for monitoring and validation."
        )

        recommendations.append(
            f"Monitor and validate data quality for high-impact features: {', '.join(top_three)}."
        )

    if target_column and target_column in df.columns:
        target_missing = df[target_column].isnull().sum()
        if target_missing > 0:
            recommendations.append(
                f"Resolve {target_missing} missing target values in '{target_column}' before production deployment."
            )

    if problem_type == "Classification" and best_score < 0.75:
        recommendations.append(
            "Consider feature engineering, class balancing, or hyperparameter tuning to improve accuracy."
        )
    elif problem_type == "Regression" and best_score < 0.6:
        recommendations.append(
            "Consider non-linear features, outlier treatment, or advanced regression models."
        )
    else:
        recommendations.append(
            f"If appropriate, consider deploying **{best_model_name}** as the primary model and establish a monitoring baseline to track performance over time."
        )

    recommendations.append(
        "Schedule periodic retraining as new data arrives to maintain model performance."
    )

    confidence = _compute_confidence(best_score, problem_type, score_spread, df, target_column)
    return insights, recommendations, confidence


def _compute_confidence(best_score, problem_type, score_spread, df, target_column):
    base = best_score if problem_type == "Classification" else min(best_score, 1.0)
    base = max(0, min(1, base))

    spread_bonus = min(0.1, score_spread * 0.5)
    size_bonus = min(0.1, len(df) / 10000)

    penalty = 0
    if target_column and target_column in df.columns:
        missing_rate = df[target_column].isnull().mean()
        penalty += missing_rate * 0.2

    confidence = min(0.99, max(0.35, base * 0.75 + spread_bonus + size_bonus - penalty))
    return round(confidence * 100, 1)


def generate_executive_summary(df, health, problem_type, best_model_name, results, confidence):
    rows, cols = df.shape if df is not None else (0, 0)
    best_score = max(results.values()) if results else 0
    metric_label = "accuracy" if problem_type == "Classification" else "R²"

    score_text = f"{best_score:.2%}" if problem_type == "Classification" else f"{best_score:.4f}"

    # Compose a clear executive summary suitable for business users
    lines = []
    lines.append(f"This report summarizes an end-to-end analysis of a dataset with {rows:,} rows and {cols} features.")
    lines.append(
        f"Overall data quality is assessed at {health['score']}/100 ({health.get('grade', 'N/A')}). "
        "This score reflects missing values, duplicates, and other common issues that can affect model training."
    )
    if results:
        lines.append(
            f"A {problem_type.lower()} pipeline was executed and the top model is {best_model_name} with {metric_label} of {score_text}. "
            f"AI confidence for this recommendation is {confidence}%."
        )
    else:
        lines.append("Modeling was not performed or no results are available yet; run AutoML Studio to produce model comparison metrics.")

    lines.append("Key findings and clear next steps are provided in the following sections for both technical and business audiences.")

    return "\n\n".join(lines)


def generate_beginner_report_sections(df, health, issues, model_info=None, eda_findings=None):
    """Create beginner-friendly executive and dataset summary text.

    Returns (executive_summary, dataset_summary)
    """
    rows, cols = df.shape if df is not None else (0, 0)
    model_info = model_info or {}
    eda_findings = eda_findings or []

    # Simple dataset summary for beginners
    dataset_summary = (
        f"This dataset has {rows:,} rows and {cols} columns. "
        "Each row is one record (for example, one customer or one house), "
        "and each column is a piece of information about that record (for example, age or price)."
    )

    # Explain data quality (health score)
    health_score = health.get("score") if isinstance(health, dict) else None
    health_grade = health.get("grade") if isinstance(health, dict) else None
    quality_sentence = ""
    if health_score is not None:
        quality_sentence = (
            f"The dataset received a quality score of {health_score}/100"
            + (f" (grade: {health_grade}). " if health_grade else ". ")
            + "This score summarizes missing values, duplicates, and other common issues."
        )

    # EDA findings — keep simple and limited
    eda_lines = []
    for f in (eda_findings or [])[:5]:
        # remove markdown emphasis if present
        eda_lines.append(str(f).replace("**", ""))

    eda_sentence = ""
    if eda_lines:
        eda_sentence = "Key patterns found: " + "; ".join(eda_lines) + "."

    # Model summary
    problem = model_info.get("problem_type") or model_info.get("problem") or "N/A"
    best_model = model_info.get("best_model") or model_info.get("best_model_name") or "N/A"
    confidence = model_info.get("confidence")
    target = model_info.get("target_column") or model_info.get("target") or "the target"

    model_sentence = (
        f"Objective: build a model to predict '{target}' using the available data. "
        f"This is a {problem.lower() if isinstance(problem, str) else 'prediction'} task. "
        f"The selected model is {best_model}."
    )

    # Explain what the model score means using model_results if available
    model_results = model_info.get("model_results") or []
    score_sentence = ""
    if model_results:
        # try to parse the best model's score from model_results list if available
        try:
            # example entry: "ModelName: 82.34%" or "ModelName: 0.8234"
            for entry in model_results:
                if best_model and str(entry).startswith(str(best_model)):
                    _, _, score = entry.partition(": ")
                    score_sentence = f"Model performance: {score}. "
                    break
        except Exception:
            score_sentence = "Model performance metrics are included in the report. "

    # Important features
    shap = model_info.get("shap_ranking") or []
    important_feats = []
    for item in shap[:5]:
        # item expected like 'feature: 0.0123'
        important_feats.append(str(item).split(":")[0])
    feat_sentence = ""
    if important_feats:
        feat_sentence = (
            "The features most affecting predictions are: "
            + ", ".join(important_feats)
            + ". These are the pieces of data the model uses most when deciding."
        )

    # Business impact, limitations, recommendations
    impact = (
        "Business impact: If accurate, this model can help automate decisions, "
        "identify high-risk cases, or support business planning by predicting outcomes."
    )

    limitations = (
        "Limitations: The model may be affected by missing or biased data, "
        "and its predictions are only as good as the data and assumptions used to train it."
    )

    recommendations = (
        "Next steps: review and fix data quality issues, collect more data if possible, "
        "try simple feature improvements, and set up monitoring to watch model performance over time."
    )

    # Compose executive summary with clear short paragraphs
    parts = []
    parts.append("<strong>Project Objective & Business Problem</strong>")
    parts.append(model_sentence)
    parts.append(quality_sentence)
    parts.append("<strong>What the dataset contains</strong>")
    parts.append(dataset_summary)
    if eda_sentence:
        parts.append("<strong>Key patterns found during exploration</strong>")
        parts.append(eda_sentence)
    parts.append("<strong>Model selection & meaning</strong>")
    parts.append(score_sentence + feat_sentence)
    parts.append("<strong>Business impact</strong>")
    parts.append(impact)
    parts.append("<strong>Limitations</strong>")
    parts.append(limitations)
    parts.append("<strong>Recommendations & Next Steps</strong>")
    parts.append(recommendations)

    # Join into HTML paragraphs for UI and PDF
    executive = "".join(f"<p>{p}</p>" for p in parts if p)

    return executive, dataset_summary + (" " + quality_sentence if quality_sentence else "")

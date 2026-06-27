"""LLM-powered insights with rule-based fallback."""

import os

import streamlit as st


def _get_api_key():
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        return None


def generate_llm_insight(prompt, fallback_fn, *args, **kwargs):
    """Try LLM first; fall back to rule-based generator."""
    api_key = _get_api_key()
    if api_key and os.environ.get("OPENAI_API_KEY"):
        try:
            return _openai_insight(prompt)
        except Exception:
            pass
    return fallback_fn(*args, **kwargs)


def _openai_insight(prompt):
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert data scientist assistant. Be concise and actionable."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
        temperature=0.4,
    )
    return response.choices[0].message.content


def generate_dataset_summary_llm(df, health, issues):
    api_key = _get_api_key()
    if not api_key:
        return _rule_dataset_summary(df, health, issues)

    prompt = (
        f"Summarize this dataset in 3 sentences for an executive audience. "
        f"Rows: {df.shape[0]}, Columns: {df.shape[1]}, "
        f"Health score: {health['score']}/100, Issues: {len(issues)}."
    )
    try:
        return generate_llm_insight(prompt, _rule_dataset_summary, df, health, issues)
    except Exception:
        return _rule_dataset_summary(df, health, issues)


def _rule_dataset_summary(df, health, issues):
    return (
        f"Your dataset contains {df.shape[0]:,} records across {df.shape[1]} features "
        f"with a quality score of {health['score']}/100 ({health['grade']}). "
        f"I detected {len(issues)} data quality item(s) requiring attention. "
        f"{health['summary']}"
    )


def generate_eda_insight_llm(df, column, stats):
    api_key = _get_api_key()
    if not api_key:
        return _rule_eda_insight(column, stats)

    prompt = (
        f"Provide one insightful sentence about the '{column}' feature: "
        f"mean={stats.get('mean', 'N/A')}, std={stats.get('std', 'N/A')}, "
        f"skew={stats.get('skew', 'N/A')}, outlier_pct={stats.get('outlier_pct', 0)}%."
    )
    try:
        return generate_llm_insight(prompt, _rule_eda_insight, column, stats)
    except Exception:
        return _rule_eda_insight(column, stats)


def _rule_eda_insight(column, stats):
    skew = stats.get("skew", 0)
    outlier_pct = stats.get("outlier_pct", 0)
    skew_desc = "right-skewed" if skew > 0.5 else "left-skewed" if skew < -0.5 else "approximately symmetric"
    outlier_note = f" and contains ~{outlier_pct:.1f}% outliers" if outlier_pct > 1 else ""
    return f"**{column}** distribution is {skew_desc}{outlier_note}."


def generate_cleaning_advice_llm(missing_pct, duplicates, issues):
    api_key = _get_api_key()
    if not api_key:
        return _rule_cleaning_advice(missing_pct, duplicates, issues)

    prompt = (
        f"As a data cleaning AI assistant, recommend actions. "
        f"Missing: {missing_pct:.1f}%, Duplicates: {duplicates}, Issues: {len(issues)}."
    )
    try:
        return generate_llm_insight(prompt, _rule_cleaning_advice, missing_pct, duplicates, issues)
    except Exception:
        return _rule_cleaning_advice(missing_pct, duplicates, issues)


def _rule_cleaning_advice(missing_pct, duplicates, issues):
    parts = []
    if missing_pct > 0:
        parts.append(
            f"Your dataset has {missing_pct:.1f}% missing values. "
            "I recommend median imputation for numerical columns and mode imputation for categorical columns."
        )
    if duplicates > 0:
        parts.append(f"I found {duplicates} duplicate records — removing them will improve model reliability.")
    if not parts:
        parts.append("Your dataset looks clean. Proceed to exploratory analysis and modeling.")
    return " ".join(parts)

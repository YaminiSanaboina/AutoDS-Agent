import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def _numeric_columns(df):
    return df.select_dtypes(include="number").columns.tolist()


def generate_eda(df):
    results = {}

    numerical_columns = df.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns

    results["numerical_columns"] = list(numerical_columns)
    results["categorical_columns"] = [
        c for c in df.columns if c not in numerical_columns
    ]
    results["summary"] = df.describe(include="all")

    return results


def plot_histogram(df, column):
    fig = px.histogram(
        df,
        x=column,
        title=f"Distribution of {column}",
        color_discrete_sequence=["#6366F1"],
    )
    fig.update_layout(height=400)
    return fig


def plot_boxplot(df, column):
    fig = px.box(
        df,
        y=column,
        title=f"Boxplot — {column}",
        color_discrete_sequence=["#8B5CF6"],
    )
    return fig


def plot_violin(df, column):
    fig = px.violin(
        df,
        y=column,
        title=f"Violin Plot — {column}",
        box=True,
        color_discrete_sequence=["#06B6D4"],
    )
    return fig


def plot_scatter(df, x_col, y_col):
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        title=f"{x_col} vs {y_col}",
        opacity=0.7,
        color_discrete_sequence=["#6366F1"],
    )
    return fig


def plot_pair(df, columns, max_cols=4):
    cols = columns[:max_cols]
    if len(cols) < 2:
        return None
    fig = px.scatter_matrix(
        df[cols],
        dimensions=cols,
        title="Pair Plot",
        opacity=0.6,
    )
    return fig


def missing_values_chart(df):
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]
    missing = missing[missing["Missing Values"] > 0]

    if missing.empty:
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Values"]

    fig = px.bar(
        missing,
        x="Column",
        y="Missing Values",
        title="Missing Values Analysis",
        color="Missing Values",
        color_continuous_scale="Reds",
    )
    return fig


def correlation_heatmap(df):
    numerical_df = df.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    )

    if numerical_df.empty or numerical_df.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough numerical columns", showarrow=False)
        return fig

    correlation = numerical_df.corr()

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        title="Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    return fig


def detect_outliers(df, column):
    series = df[column].dropna()
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return {
        "count": len(outliers),
        "pct": len(outliers) / max(len(series), 1) * 100,
        "lower": lower,
        "upper": upper,
    }


def compute_feature_stats(df, column):
    series = df[column].dropna()
    if series.empty:
        return {}
    outlier_info = detect_outliers(df, column)
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "skew": float(series.skew()),
        "outlier_pct": outlier_info["pct"],
        "outlier_count": outlier_info["count"],
    }


def generate_eda_insights(df, numerical_columns):
    insights = []
    for col in numerical_columns[:5]:
        stats = compute_feature_stats(df, col)
        if not stats:
            continue
        skew = stats["skew"]
        skew_desc = "right-skewed" if skew > 0.5 else "left-skewed" if skew < -0.5 else "symmetric"
        outlier_note = ""
        if stats["outlier_pct"] > 1:
            outlier_note = f" and contains ~{stats['outlier_pct']:.1f}% outliers"
        insights.append(f"**{col}** distribution is {skew_desc}{outlier_note}.")
    return insights
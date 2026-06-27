"""Specialized AI agents for AutoDS Agent.

Each agent returns structured dictionary responses for communication with a
supervisory AISupervisor.
"""
from __future__ import annotations

from typing import Any, Dict, List
import streamlit as st
import pandas as pd
import numpy as np

from utils.session_manager import SessionKeys, get_dataframe


class BaseAgent:
    """Common response format for all agents."""

    agent_name: str = "Base Agent"

    def _response(self, status: str, confidence: float, summary: str, recommendations: List[str]) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "status": status,
            "confidence": round(float(confidence), 2),
            "summary": summary,
            "recommendations": recommendations,
        }


class DataAgent(BaseAgent):
    agent_name = "Data Agent"

    def analyze_dataset(self) -> Dict[str, Any]:
        df = get_dataframe()
        if df is None or df.empty:
            return self._response("Incomplete", 0.0, "No dataset is loaded.", ["Upload a dataset in Data Hub."])

        missing = int(df.isnull().sum().sum())
        duplicates = int(df.duplicated().sum())
        types = df.dtypes.apply(lambda d: d.name).to_dict()
        rows, cols = df.shape

        summary = (
            f"The dataset contains {rows} rows and {cols} columns. "
            f"There are {missing} missing values and {duplicates} duplicate rows. "
            f"Data types include {', '.join(sorted(set(types.values())))}."
        )

        recommendations: List[str] = []
        if missing > 0:
            recommendations.append("Impute or remove missing values based on feature type.")
        if duplicates > 0:
            recommendations.append("Remove or investigate duplicate rows.")
        if rows < 100:
            recommendations.append("Dataset is small; plan to use cross-validation.")
        if cols > rows:
            recommendations.append("Perform feature selection because there are more columns than rows.")
        if not recommendations:
            recommendations.append("Dataset quality appears adequate for exploration.")

        return self._response("Complete", 0.86, summary, recommendations)

    def suggest_cleaning(self) -> Dict[str, Any]:
        df = get_dataframe()
        if df is None or df.empty:
            return self._response("Incomplete", 0.0, "No dataset is loaded.", ["Upload a dataset first."])

        missing = int(df.isnull().sum().sum())
        duplicates = int(df.duplicated().sum())
        categorical = df.select_dtypes(include=["object"]).columns.tolist()
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()

        summary = "Suggested cleaning strategy generated based on dataset characteristics."
        recommendations: List[str] = []
        if missing > 0:
            if numeric:
                recommendations.append("Impute numeric missing values with median or KNN imputation.")
            if categorical:
                recommendations.append("Impute categorical missing values with mode or constant label.")
        if duplicates > 0:
            recommendations.append("Remove duplicate rows after verifying data integrity.")
        if not recommendations:
            recommendations.append("No major cleaning actions detected. Proceed to EDA.")

        return self._response("Complete", 0.82, summary, recommendations)


class EDAAgent(BaseAgent):
    agent_name = "EDA Agent"

    def analyze_patterns(self) -> Dict[str, Any]:
        df = get_dataframe()
        if df is None or df.empty:
            return self._response("Incomplete", 0.0, "No dataset loaded for EDA.", ["Upload a dataset first."])

        numeric = df.select_dtypes(include=[np.number])
        correlations = numeric.corr().abs().fillna(0)
        max_corr = 0.0
        max_pair = (None, None)
        for col in correlations.columns:
            for row in correlations.index:
                if col != row and correlations.loc[row, col] > max_corr:
                    max_corr = correlations.loc[row, col]
                    max_pair = (row, col)

        summary = "Exploratory data analysis is ready. "
        if max_pair[0] and max_corr > 0.7:
            summary += f"Strong correlation found between {max_pair[0]} and {max_pair[1]} (r={max_corr:.2f})."
        else:
            summary += "No exceptionally strong linear correlations were detected among numeric features."

        recommendations: List[str] = ["Review feature distributions and correlations in EDA Explorer."]
        if max_corr > 0.7:
            recommendations.append(f"Investigate whether {max_pair[0]} and {max_pair[1]} are redundant.")

        return self._response("Complete", 0.78, summary, recommendations)

    def generate_insights(self) -> Dict[str, Any]:
        df = get_dataframe()
        if df is None or df.empty:
            return self._response("Incomplete", 0.0, "No dataset loaded for generating insights.", ["Upload a dataset first."])

        numeric = df.select_dtypes(include=[np.number])
        insights: List[str] = []
        if not numeric.empty:
            desc = numeric.describe().loc[["mean", "std"]]
            insights.append("Numeric features show central tendencies and dispersion.")
            if (desc.loc["std"] / desc.loc["mean"]).max() > 1:
                insights.append("Some numeric features have high relative variance and may need scaling.")
        else:
            insights.append("No numeric features available for distribution analysis.")

        summary = "EDA insights generated for the current dataset."
        recommendations = ["Identify feature relationships and outliers."]
        return self._response("Complete", 0.75, summary, recommendations)


class MLAgent(BaseAgent):
    agent_name = "ML Agent"

    def evaluate_models(self) -> Dict[str, Any]:
        if not st.session_state.get(SessionKeys.MODEL_TRAINED):
            return self._response("Incomplete", 0.0, "No trained models available.", ["Train models in AutoML Studio."])

        best = st.session_state.get(SessionKeys.BEST_MODEL_NAME)
        results = st.session_state.get(SessionKeys.RESULTS) or {}
        if best and best in results:
            score = results[best]
            summary = f"Best model {best} has a performance score of {score}."
            recommendations = ["Compare model metrics across trained algorithms."]
            return self._response("Complete", 0.88, summary, recommendations)

        return self._response("Incomplete", 0.6, "Model training completed but best model details are missing.", ["Review AutoML results."])

    def suggest_models(self) -> Dict[str, Any]:
        problem = st.session_state.get(SessionKeys.PROBLEM_TYPE)
        recs: List[str] = []
        if problem == "Classification":
            recs = ["Consider Random Forest for robustness.", "Use Logistic Regression as a strong baseline.", "Try XGBoost for potential accuracy improvements."]
        elif problem == "Regression":
            recs = ["Consider Linear Regression for interpretability.", "Use Random Forest Regressor for non-linear relationships.", "Try Gradient Boosting for improved accuracy."]
        else:
            recs = ["Train models in AutoML Studio to determine the best approach."]

        summary = "Model suggestions based on the identified problem type."
        confidence = 0.8 if problem else 0.5
        return self._response("Complete", confidence, summary, recs)


class XAIAgent(BaseAgent):
    agent_name = "XAI Agent"

    def explain_model(self) -> Dict[str, Any]:
        if not st.session_state.get(SessionKeys.SHAP_COMPUTED) and not st.session_state.get(SessionKeys.SHAP_VALUES):
            return self._response("Incomplete", 0.0, "SHAP explanations are not available.", ["Generate XAI explanations in AI Decision Intelligence."])

        summary = "SHAP-based explainability is available for model interpretation."
        recommendations = ["Review top feature impacts and explain them to stakeholders."]
        return self._response("Complete", 0.85, summary, recommendations)


class ReportAgent(BaseAgent):
    agent_name = "Report Agent"

    def review_report(self) -> Dict[str, Any]:
        if not st.session_state.get(SessionKeys.REPORT_GENERATED):
            return self._response("Incomplete", 0.0, "No report has been generated yet.", ["Generate the AI Research Report."])

        summary = "A report has been generated and can be reviewed for completeness."
        recommendations = ["Add additional sections on model risks and business impact if needed."]
        return self._response("Complete", 0.82, summary, recommendations)

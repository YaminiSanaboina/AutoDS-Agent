"""AISupervisor for AutoDS Agent.

This module coordinates specialized agents and provides an advisory brain layer
for the existing AutoDS backend.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import streamlit as st

from utils.session_manager import SessionKeys, get_dataframe
from agents.specialized_agents import DataAgent, EDAAgent, MLAgent, XAIAgent, ReportAgent


class AISupervisor:
    def __init__(self) -> None:
        self.data_agent = DataAgent()
        self.eda_agent = EDAAgent()
        self.ml_agent = MLAgent()
        self.xai_agent = XAIAgent()
        self.report_agent = ReportAgent()
        self.decision_history: List[Dict[str, Any]] = []

    def analyze_project_state(self) -> Dict[str, Any]:
        df = get_dataframe()
        state_summary: List[str] = []
        state_tags: List[str] = []

        if df is None or df.empty:
            state_summary.append("No dataset loaded.")
            state_tags.append("dataset_pending")
        else:
            state_summary.append(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
            state_tags.append("dataset_ready")

        if st.session_state.get(SessionKeys.CLEANING_REPORT) or (st.session_state.get(SessionKeys.CLEANING_HISTORY) or []):
            state_summary.append("Cleaning stage completed.")
            state_tags.append("cleaning_done")
        else:
            state_summary.append("Cleaning stage pending.")
            state_tags.append("cleaning_pending")

        if st.session_state.get(SessionKeys.EDA_GENERATED):
            state_summary.append("EDA completed.")
            state_tags.append("eda_done")
        else:
            state_summary.append("EDA stage pending.")
            state_tags.append("eda_pending")

        if st.session_state.get(SessionKeys.MODEL_TRAINED):
            state_summary.append("Model training completed.")
            state_tags.append("ml_done")
        else:
            state_summary.append("Model training not completed.")
            state_tags.append("ml_pending")

        if st.session_state.get(SessionKeys.SHAP_COMPUTED):
            state_summary.append("Explainability analysis available.")
            state_tags.append("xai_ready")
        else:
            state_summary.append("Explainability analysis pending.")
            state_tags.append("xai_pending")

        if st.session_state.get(SessionKeys.REPORT_GENERATED):
            state_summary.append("Report generation completed.")
            state_tags.append("report_ready")
        else:
            state_summary.append("Report generation not started.")
            state_tags.append("report_pending")

        return {
            "summary": " ".join(state_summary),
            "tags": state_tags,
            "dataset_loaded": bool(df is not None and not df.empty),
            "cleaning_done": bool(st.session_state.get(SessionKeys.CLEANING_REPORT)) or bool(st.session_state.get(SessionKeys.CLEANING_HISTORY)),
            "eda_done": bool(st.session_state.get(SessionKeys.EDA_GENERATED)),
            "model_trained": bool(st.session_state.get(SessionKeys.MODEL_TRAINED)),
            "xai_ready": bool(st.session_state.get(SessionKeys.SHAP_COMPUTED)),
            "report_ready": bool(st.session_state.get(SessionKeys.REPORT_GENERATED)),
        }

    def recommend_next_agent(self) -> Dict[str, Any]:
        project = self.analyze_project_state()
        if not project["dataset_loaded"]:
            return {"next_agent": "DataAgent", "reason": "Load a dataset before proceeding.", "recommendations": ["Upload or import a dataset in Data Hub."]}

        if not project["cleaning_done"]:
            return {"next_agent": "DataAgent", "reason": "Dataset cleaning is the highest priority.", "recommendations": ["Run data cleaning tasks before EDA."]}

        if not project["eda_done"]:
            return {"next_agent": "EDAAgent", "reason": "Exploratory analysis should happen after cleaning.", "recommendations": ["Complete EDA to understand feature relationships."]}

        if not project["model_trained"]:
            return {"next_agent": "MLAgent", "reason": "Training models is the core next step.", "recommendations": ["Use AutoML Studio to train models."]}

        if not project["xai_ready"]:
            return {"next_agent": "XAIAgent", "reason": "Explainability should be evaluated after model training.", "recommendations": ["Generate SHAP explanations for the trained model."]}

        if not project["report_ready"]:
            return {"next_agent": "ReportAgent", "reason": "Create a report to summarize results.", "recommendations": ["Generate the AI Research Report."]}

        return {"next_agent": "AISupervisor", "reason": "All major stages are complete.", "recommendations": ["Review outputs and refine the model or report as needed."]}

    def delegate_task(self, task_name: str) -> Dict[str, Any]:
        dispatch = {
            "DataAgent.analyze_dataset": self.data_agent.analyze_dataset,
            "DataAgent.suggest_cleaning": self.data_agent.suggest_cleaning,
            "EDAAgent.analyze_patterns": self.eda_agent.analyze_patterns,
            "EDAAgent.generate_insights": self.eda_agent.generate_insights,
            "MLAgent.evaluate_models": self.ml_agent.evaluate_models,
            "MLAgent.suggest_models": self.ml_agent.suggest_models,
            "XAIAgent.explain_model": self.xai_agent.explain_model,
            "ReportAgent.review_report": self.report_agent.review_report,
        }

        action = dispatch.get(task_name)
        if not action:
            return {
                "status": "Failed",
                "summary": f"Task {task_name} is not supported.",
                "recommendations": ["Use a supported agent task name."],
            }

        result = action()
        self.add_decision_log(task_name, result)
        return result

    def add_decision_log(self, task_name: str, result: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            "task_name": task_name,
            "result": result or {},
            "project_state": self.analyze_project_state(),
        }
        self.decision_history.append(entry)

    def generate_project_intelligence(self) -> Dict[str, Any]:
        project = self.analyze_project_state()
        next_step = self.recommend_next_agent()
        summaries: List[str] = [project["summary"], next_step["reason"]]
        recommendations = next_step["recommendations"]

        if project["dataset_loaded"]:
            data_summary = self.data_agent.analyze_dataset()
            summaries.append(data_summary["summary"])
            recommendations.extend(data_summary["recommendations"][:2])

        return {
            "overall_status": "Advisory",
            "project_summary": " ".join(summaries),
            "next_agent": next_step["next_agent"],
            "recommendations": recommendations,
            "decision_history": self.decision_history,
        }

"""AI Workflow Agent for AutoDS Agent.

Provides a lightweight autonomous workflow tracker that inspects Streamlit
session state (via SessionKeys) and summarizes the ML pipeline, generates
human-readable decision logs, recommends next steps, and stores a session
history of workflow events.

This module only reads and writes session state and does not modify AutoML,
SHAP, reports, or UI pages.
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from utils.session_manager import SessionKeys, has_dataset


class AIWorkflowAgent:
    """Agent that inspects Streamlit session state and guides the user.

    Methods are designed to be called from UI pages but do not perform any
    UI rendering themselves.
    """

    def __init__(self) -> None:
        # ensure history exists
        # initialize workflow history safely
        st.session_state.setdefault("workflow_history", [])

    def analyze_workflow(self) -> Dict[str, Any]:
        """Inspect session state and return structured workflow status."""
        s = st.session_state

        dataset_loaded = has_dataset()
        dataset_analyzed = bool(s.get(SessionKeys.EDA_GENERATED) or s.get(SessionKeys.DATASET_METADATA))

        # Data quality: consider cleaning done when a cleaning report or history exists
        quality_checked = bool(s.get(SessionKeys.CLEANING_REPORT) or (s.get(SessionKeys.CLEANING_HISTORY) or []))
        cleaned = bool(s.get(SessionKeys.HEALTH_AFTER)) or bool(s.get(SessionKeys.CLEANING_HISTORY))

        eda_done = bool(s.get(SessionKeys.EDA_GENERATED))

        model_trained = bool(s.get(SessionKeys.MODEL_TRAINED))
        best_model_selected = bool(s.get(SessionKeys.BEST_MODEL_NAME) and s.get(SessionKeys.BEST_MODEL))

        shap_done = bool(s.get(SessionKeys.SHAP_COMPUTED) or s.get(SessionKeys.SHAP_VALUES) or s.get(SessionKeys.SHAP_IMPORTANCE))

        predictions_made = bool(s.get("prediction_history") or s.get("last_prediction"))

        report_generated = bool(s.get(SessionKeys.REPORT_GENERATED))

        completed_steps = []
        pending_steps = []

        # Determine completed/pending steps
        if not dataset_loaded:
            current_stage = "no_dataset_loaded"
            pending_steps.extend(["Upload dataset", "Check quality", "Run EDA", "Train model", "Explain model", "Predict & Report"])
        else:
            # dataset present
            current_stage = "dataset_loaded"
            completed_steps.append("Dataset loaded")
            if not quality_checked:
                pending_steps.append("Check data quality")
            else:
                completed_steps.append("Data quality checked")

            if not eda_done:
                pending_steps.append("Run EDA")
            else:
                completed_steps.append("EDA completed")

            if not model_trained:
                pending_steps.append("Train models in AutoML Studio")
            else:
                completed_steps.append("Models trained")
                if not best_model_selected:
                    pending_steps.append("Select best model")
                else:
                    completed_steps.append("Best model selected")

            if not shap_done:
                pending_steps.append("Generate SHAP explanations")
            else:
                completed_steps.append("SHAP explanations generated")

            if not predictions_made:
                pending_steps.append("Create predictions")
            else:
                completed_steps.append("Predictions performed")

            if not report_generated:
                pending_steps.append("Generate AI Research Report")
            else:
                completed_steps.append("Report generated")

            # choose a best guess for current stage
            if report_generated:
                current_stage = "report_generated"
            elif predictions_made:
                current_stage = "predictions_performed"
            elif shap_done:
                current_stage = "explainability_completed"
            elif model_trained:
                current_stage = "model_trained"
            elif eda_done:
                current_stage = "eda_completed"
            elif quality_checked:
                current_stage = "quality_checked"

        progress = self.calculate_progress()

        return {
            "current_stage": current_stage,
            "completed_steps": completed_steps,
            "pending_steps": pending_steps,
            "progress_percentage": progress["percentage"],
            "progress_label": progress["label"],
        }

    def generate_decision_log(self) -> List[str]:
        """Create human-readable decision log entries based on session state."""
        s = st.session_state
        logs: List[str] = []

        # Dataset
        name = s.get(SessionKeys.DATASET_NAME) or s.get(SessionKeys.UPLOAD_FILENAME) or "Untitled Dataset"
        md = s.get(SessionKeys.DATASET_METADATA) or {}
        rows = md.get("rows") if isinstance(md, dict) else None
        cols = md.get("columns") if isinstance(md, dict) else None
        if has_dataset():
            r = f"AI detected a dataset named '{name}'"
            if rows is not None and cols is not None:
                r += f" with {rows} records and {cols} features."
            else:
                r += "."
            logs.append(r)

        # Data quality
        if s.get(SessionKeys.CLEANING_REPORT) or (s.get(SessionKeys.CLEANING_HISTORY) or []):
            cr = s.get(SessionKeys.CLEANING_REPORT)
            # Try to pick simple examples from cleaning report or metadata
            missing_total = None
            if isinstance(md, dict):
                missing_total = md.get("missing_total")
            if missing_total:
                logs.append(f"AI found {missing_total} missing values across the dataset. Recommended targeted imputation or removal depending on the feature semantics.")
            else:
                logs.append("Data quality checks were performed. Review the cleaning report for details and suggested actions.")

        # EDA
        if s.get(SessionKeys.EDA_GENERATED) or (s.get(SessionKeys.EDA_SUMMARY) is not None):
            logs.append("EDA completed: patterns and basic distributions were identified. Check EDA Explorer for visuals and insights.")

        # AutoML
        if s.get(SessionKeys.MODEL_TRAINED):
            best = s.get(SessionKeys.BEST_MODEL_NAME)
            results = s.get(SessionKeys.RESULTS) or {}
            if best and best in results:
                score = results[best]
                logs.append(f"AutoML completed. Best model '{best}' achieved a score of {score}.")
            elif best:
                logs.append(f"AutoML completed. Best model '{best}' has been selected.")
            else:
                logs.append("AutoML completed. Models were trained and evaluated; select the best model.")

        # SHAP
        if s.get(SessionKeys.SHAP_COMPUTED) or s.get(SessionKeys.SHAP_VALUES) or s.get(SessionKeys.SHAP_IMPORTANCE):
            fi = s.get(SessionKeys.SHAP_IMPORTANCE)
            if fi and isinstance(fi, dict):
                top = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                feats = ", ".join([t[0] for t in top])
                logs.append(f"Feature importance analysis identified {feats} as the strongest predictors.")
            else:
                logs.append("SHAP-based explainability was generated for model interpretation.")

        # Predictions
        if s.get("prediction_history") or s.get("last_prediction"):
            last = None
            if s.get("last_prediction"):
                last = s.get("last_prediction")
            else:
                hist = s.get("prediction_history") or []
                last = hist[0] if hist else None
            if last:
                logs.append(f"Predictions were produced. The most recent prediction: {last.get('prediction')}.")
            else:
                logs.append("Predictions were produced for example inputs.")

        # Reports
        if s.get(SessionKeys.REPORT_GENERATED):
            logs.append("A final AI Research Report has been generated and is available for download.")

        return logs

    def recommend_next_action(self) -> str:
        """Provide a short, actionable next-step recommendation."""
        s = st.session_state

        if not has_dataset():
            return "Start by uploading a dataset in Data Hub."

        # dataset loaded but quality not checked
        quality_checked = bool(s.get(SessionKeys.CLEANING_REPORT) or (s.get(SessionKeys.CLEANING_HISTORY) or []))
        if not quality_checked:
            return "Your dataset is loaded. Check Data Quality Lab to identify missing values and duplicates."

        if not s.get(SessionKeys.EDA_GENERATED):
            return "Run EDA in EDA Explorer to discover distributions and patterns in your data."

        if not s.get(SessionKeys.MODEL_TRAINED):
            return "Proceed to AutoML Studio to train models on your selected target."

        if s.get(SessionKeys.MODEL_TRAINED) and not (s.get(SessionKeys.SHAP_COMPUTED) or s.get(SessionKeys.SHAP_VALUES)):
            return "Your model is trained. Generate SHAP explanations in AI Decision Intelligence to inspect feature impacts."

        if s.get(SessionKeys.MODEL_TRAINED) and (s.get(SessionKeys.SHAP_COMPUTED) or s.get(SessionKeys.SHAP_VALUES)) and not (s.get("prediction_history") or s.get("last_prediction")):
            return "Run the Prediction Playground to test the model on new examples and perform what-if analysis."

        if s.get(SessionKeys.REPORT_GENERATED):
            return "Workflow complete. Review or download the AI Research Report."

        # Everything else
        return "Congratulations. Your machine learning workflow is complete. Generate a final AI Research Report."

    def calculate_progress(self) -> Dict[str, Any]:
        """Return progress percentage and a status label."""
        s = st.session_state

        dataset_loaded = has_dataset()
        cleaned = bool(s.get(SessionKeys.HEALTH_AFTER) or (s.get(SessionKeys.CLEANING_HISTORY) or []))
        eda_done = bool(s.get(SessionKeys.EDA_GENERATED))
        model_trained = bool(s.get(SessionKeys.MODEL_TRAINED))
        shap_done = bool(s.get(SessionKeys.SHAP_COMPUTED) or s.get(SessionKeys.SHAP_VALUES))
        predictions = bool(s.get("prediction_history") or s.get("last_prediction"))
        report_done = bool(s.get(SessionKeys.REPORT_GENERATED))

        # Follow the example mapping: 0,25,50,75,100
        if not dataset_loaded:
            percentage = 0
            label = "Beginner"
        else:
            # require cleaning for the first milestone
            percentage = 0
            if dataset_loaded and cleaned:
                percentage = 25
            if eda_done:
                percentage = 50
            if model_trained and shap_done:
                percentage = 75
            if predictions and report_done:
                percentage = 100

            if percentage == 25:
                label = "Exploring Data"
            elif percentage == 50:
                label = "Building Models"
            elif percentage == 75:
                label = "AI Expert"
            elif percentage == 100:
                label = "AI Expert"
            else:
                label = "Beginner"

        return {"percentage": percentage, "label": label}

    def get_workflow_summary(self) -> Dict[str, Any]:
        """Return a detailed summary including dataset health and model info."""
        s = st.session_state

        wf = self.analyze_workflow()

        md = s.get(SessionKeys.DATASET_METADATA) or {}
        dataset_info = {
            "name": s.get(SessionKeys.DATASET_NAME) or s.get(SessionKeys.UPLOAD_FILENAME) or "Untitled Dataset",
            "rows": md.get("rows") if isinstance(md, dict) else None,
            "columns": md.get("columns") if isinstance(md, dict) else None,
            "problem_type": s.get(SessionKeys.PROBLEM_TYPE),
            "quality_score": None,
        }
        # try to surface a quality/health score
        if isinstance(md, dict) and md.get("health") is not None:
            dataset_info["quality_score"] = md.get("health")

        best_model = s.get(SessionKeys.BEST_MODEL_NAME)
        results = s.get(SessionKeys.RESULTS) or {}
        model_info = None
        if best_model:
            model_info = {"name": best_model, "score": results.get(best_model)}

        return {
            "workflow": wf,
            "dataset": dataset_info,
            "model": model_info,
            "recent_events": list(s.get("workflow_history") or []),
        }

    def add_workflow_event(self, action: str, explanation: Optional[str] = None, timestamp: Optional[str] = None) -> None:
        """Append an event to `st.session_state['workflow_history']` keeping latest 25 events."""
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        entry = {"timestamp": timestamp, "action": action, "explanation": explanation}
        hist = st.session_state.get("workflow_history") or []
        hist.insert(0, entry)
        st.session_state["workflow_history"] = hist[:25]

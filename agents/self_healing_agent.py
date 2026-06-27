"""Self-healing AutoML agent for AutoDS Agent.

This backend-only module analyzes ML failures, suggests safe fixes, stores an
error history, and provides recovery guidance without modifying datasets or UI.
"""
from __future__ import annotations

import datetime
from collections import Counter
from typing import Any, Dict, List, Optional

# In-memory history for worker-thread safety (no Streamlit in background threads).
_ERROR_HISTORY: List[Dict[str, Any]] = []


def _streamlit_context_available() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _read_history() -> List[Dict[str, Any]]:
    if _streamlit_context_available():
        import streamlit as st

        session_history = st.session_state.get(SelfHealingAgent.HISTORY_KEY)
        if isinstance(session_history, list) and session_history:
            return session_history
    return list(_ERROR_HISTORY)


def _append_history(entry: Dict[str, Any]) -> None:
    global _ERROR_HISTORY
    _ERROR_HISTORY.append(entry)
    if len(_ERROR_HISTORY) > 100:
        _ERROR_HISTORY = _ERROR_HISTORY[-100:]

    if not _streamlit_context_available():
        return

    try:
        import streamlit as st

        history = st.session_state.setdefault(SelfHealingAgent.HISTORY_KEY, [])
        history.append(entry)
        if len(history) > 100:
            st.session_state[SelfHealingAgent.HISTORY_KEY] = history[-100:]
    except Exception:
        pass


class SelfHealingAgent:
    """Agent that detects AutoML failures and recommends safe recovery actions."""

    HISTORY_KEY = "auto_ml_error_history"

    def __init__(self) -> None:
        return

    def analyze_error(self, error_message: str, dataset_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze a machine learning error and identify the root cause."""
        message = (error_message or "").strip()
        normalized = message.lower()
        analysis: Dict[str, Any] = {
            "error_type": "Unknown Error",
            "severity": "Low",
            "root_cause": "Unable to determine the root cause from the error message.",
            "confidence": 0.5,
        }

        if dataset_info and isinstance(dataset_info, dict):
            train_score = dataset_info.get("training_score")
            test_score = dataset_info.get("test_score")
            if self._detect_overfitting_from_metrics(train_score, test_score):
                analysis = {
                    "error_type": "Overfitting",
                    "severity": "Medium",
                    "root_cause": "Training performance is much higher than test performance, indicating the model is memorizing training data.",
                    "confidence": 0.92,
                }
                fix = self.recommend_fix(analysis)
                self._log_failure(message, analysis, fix)
                return analysis

        if "could not convert string" in normalized or "object dtype" in normalized:
            analysis = {
                "error_type": "Encoding Error",
                "severity": "Medium",
                "root_cause": "Categorical features were passed to a numerical model without encoding.",
                "confidence": 0.95,
            }
        elif "nan" in normalized or "missing values" in normalized or "missing value" in normalized or "null values" in normalized:
            analysis = {
                "error_type": "Missing Value Error",
                "severity": "Medium",
                "root_cause": "The model encountered missing values in the dataset during training or preprocessing.",
                "confidence": 0.92,
            }
        elif any(keyword in normalized for keyword in ["only one class", "insufficient classes", "need at least 2 classes", "must have at least 2 classes", "class imbalance"]):
            analysis = {
                "error_type": "Class Imbalance",
                "severity": "Medium",
                "root_cause": "The label distribution is skewed or contains too few classes for reliable training.",
                "confidence": 0.88,
            }
        elif "memoryerror" in normalized or "out of memory" in normalized or "memory error" in normalized:
            analysis = {
                "error_type": "Memory Error",
                "severity": "High",
                "root_cause": "The training process exceeded available memory and could not complete.",
                "confidence": 0.97,
            }
        elif any(keyword in normalized for keyword in ["inconsistent numbers of samples", "inconsistent number of samples", "feature mismatch", "shapes", "shape mismatch", "found input variables with inconsistent numbers of samples"]):
            analysis = {
                "error_type": "Shape Mismatch",
                "severity": "Medium",
                "root_cause": "The feature and label arrays do not have aligned sample counts or expected shapes.",
                "confidence": 0.9,
            }
        elif "training score" in normalized and "test score" in normalized:
            analysis = {
                "error_type": "Overfitting",
                "severity": "Medium",
                "root_cause": "Training score is significantly higher than test score, indicating a model that is overfitting.",
                "confidence": 0.9,
            }

        fix = self.recommend_fix(analysis)
        self._log_failure(message, analysis, fix)
        return analysis

    def recommend_fix(self, error_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend a safe fix based on the analyzed error."""
        error_type = error_analysis.get("error_type", "Unknown Error")
        if error_type == "Encoding Error":
            return {
                "recommended_action": "Apply one-hot encoding or label encoding to categorical features.",
                "automation_possible": True,
                "risk_level": "Low",
                "explanation": "Encoding categorical variables is safe before model training and prevents numeric-only models from failing.",
            }
        if error_type == "Missing Value Error":
            return {
                "recommended_action": "Impute missing values using mean, median, or mode imputation based on feature type.",
                "automation_possible": True,
                "risk_level": "Low",
                "explanation": "Handling missing data before training is a common and safe preprocessing step.",
            }
        if error_type == "Class Imbalance":
            return {
                "recommended_action": "Use resampling strategies such as oversampling, undersampling, or collect more data for minority classes.",
                "automation_possible": False,
                "risk_level": "Medium",
                "explanation": "Class imbalance may require careful resampling or data collection to avoid biased model behavior.",
            }
        if error_type == "Memory Error":
            return {
                "recommended_action": "Reduce dataset size, use a more memory-efficient algorithm, or increase available resources.",
                "automation_possible": False,
                "risk_level": "Medium",
                "explanation": "Memory errors usually require changing the training configuration or dataset size rather than an automatic edit.",
            }
        if error_type == "Shape Mismatch":
            return {
                "recommended_action": "Validate training and test alignment and ensure feature arrays have consistent sample counts.",
                "automation_possible": False,
                "risk_level": "Medium",
                "explanation": "Shape mismatch is typically caused by misaligned feature/label arrays and should be corrected before training.",
            }
        if error_type == "Overfitting":
            return {
                "recommended_action": "Use cross-validation, regularization, or reduce model complexity to improve generalization.",
                "automation_possible": True,
                "risk_level": "Low",
                "explanation": "Overfitting is addressed by making the model simpler or validating more rigorously, which is generally safe.",
            }
        return {
            "recommended_action": "Review the error context and dataset preprocessing to identify the root cause.",
            "automation_possible": False,
            "risk_level": "Medium",
            "explanation": "The issue could not be classified automatically; inspect preprocessing and model inputs.",
        }

    def _detect_overfitting_from_metrics(self, train_score: Any, test_score: Any) -> bool:
        try:
            if train_score is None or test_score is None:
                return False
            train = float(train_score)
            test = float(test_score)
            return train - test >= 0.15 and train >= 0.8
        except (TypeError, ValueError):
            return False

    def _log_failure(self, error_message: str, diagnosis: Dict[str, Any], fix: Dict[str, Any], success: bool = False) -> None:
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        entry = {
            "timestamp": timestamp,
            "error": error_message,
            "diagnosis": diagnosis,
            "fix": fix,
            "success": success,
        }
        _append_history(entry)

    def generate_health_report(self) -> Dict[str, Any]:
        """Generate an AutoML health report from the stored error history."""
        history: List[Dict[str, Any]] = _read_history()
        total_failures = len(history)
        resolved = sum(1 for item in history if item.get("success") is True)
        success_rate = int(round((resolved / total_failures) * 100)) if total_failures else 0
        most_common_issue = "None"
        if total_failures:
            issue_counts = Counter(item.get("diagnosis", {}).get("error_type", "Unknown Error") for item in history)
            most_common_issue = issue_counts.most_common(1)[0][0] if issue_counts else "None"

        return {
            "total_failures": total_failures,
            "resolved": resolved,
            "success_rate": success_rate,
            "most_common_issue": most_common_issue,
        }

    def simulate_fix(self) -> Dict[str, Any]:
        """Describe a safe fix preview without modifying data."""
        history: List[Dict[str, Any]] = _read_history()
        if not history:
            return {
                "fix_preview": "No error history is available to simulate a fix.",
                "requires_user_confirmation": True,
            }

        last = history[-1]
        fix_action = last.get("fix", {}).get("recommended_action", "Apply a safe preprocessing step.")
        preview = f"{last.get('diagnosis', {}).get('root_cause', '')} Recommended action: {fix_action}"
        return {
            "fix_preview": preview,
            "requires_user_confirmation": True,
        }

    def get_recovery_plan(self) -> Dict[str, Any]:
        """Return a structured recovery plan based on the latest failure."""
        history: List[Dict[str, Any]] = _read_history()
        if not history:
            return {
                "problem": "No recent AutoML failures recorded.",
                "steps": [
                    "Review the AutoML configuration.",
                    "Inspect dataset preprocessing.",
                    "Retry training with safe settings.",
                ],
            }

        last = history[-1]
        error_type = last.get("diagnosis", {}).get("error_type", "Unknown Error")
        plan_steps = self._build_recovery_steps(error_type)
        return {
            "problem": error_type,
            "steps": plan_steps,
        }

    def _build_recovery_steps(self, error_type: str) -> List[str]:
        mapping = {
            "Encoding Error": [
                "Inspect categorical features and data types.",
                "Encode categorical variables using one-hot or label encoding.",
                "Retry AutoML training.",
                "Compare results with the previous run.",
            ],
            "Missing Value Error": [
                "Identify columns with missing values.",
                "Impute missing values with mean, median, or mode.",
                "Retry AutoML training.",
                "Validate that no NaNs remain.",
            ],
            "Class Imbalance": [
                "Review the target class distribution.",
                "Apply oversampling or undersampling to balance classes.",
                "Retrain the model with the balanced dataset.",
                "Compare metrics across classes.",
            ],
            "Memory Error": [
                "Reduce dataset size or use a more efficient algorithm.",
                "Limit feature dimensionality or batch processing.",
                "Retry training with lower memory settings.",
                "Monitor memory usage during training.",
            ],
            "Shape Mismatch": [
                "Validate that training and test samples align.",
                "Ensure feature arrays have consistent shapes.",
                "Correct any preprocessing or splitting issues.",
                "Retry the AutoML pipeline.",
            ],
            "Overfitting": [
                "Use cross-validation to validate model performance.",
                "Apply regularization or simplify the model.",
                "Reduce feature complexity where possible.",
                "Compare training and test metrics again.",
            ],
        }
        return mapping.get(error_type, [
            "Review the last error diagnosis.",
            "Inspect dataset preprocessing and model inputs.",
            "Retry training with conservative settings.",
        ])

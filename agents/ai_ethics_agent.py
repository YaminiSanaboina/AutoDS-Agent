from __future__ import annotations

import pandas as pd
from typing import Any, Dict, List, Optional


class AIEthicsAgent:
    SENSITIVE_KEYS = [
        "gender",
        "sex",
        "male",
        "female",
        "age",
        "race",
        "ethnicity",
        "religion",
        "nationality",
        "marital_status",
    ]
    PRIVACY_KEYS = [
        "email",
        "phone",
        "mobile",
        "address",
        "ssn",
        "passport",
        "credit_card",
        "account_number",
    ]

    def analyze_dataset_bias(self, df: pd.DataFrame, sensitive_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        sensitive_columns = sensitive_columns or []
        detected = []
        issues = []
        recommendations = []

        lower_columns = [col.lower() for col in df.columns]
        for keyword in self.SENSITIVE_KEYS:
            for col, lower in zip(df.columns, lower_columns):
                if keyword in lower:
                    detected.append(col)

        if sensitive_columns:
            for col in sensitive_columns:
                if col in df.columns and col not in detected:
                    detected.append(col)

        if not detected:
            bias_risk = "Low"
            return {
                "bias_risk": bias_risk,
                "sensitive_features": detected,
                "issues": ["No sensitive attributes explicitly detected."],
                "recommendations": ["Continue monitoring dataset composition."],
            }

        for col in detected:
            series = df[col]
            missing_pct = float(series.isna().mean() * 100)
            counts = series.value_counts(dropna=False)
            if missing_pct > 10:
                issues.append(f"Sensitive column '{col}' has {missing_pct:.1f}% missing values.")
            if len(counts) > 0:
                lowest = counts.min()
                highest = counts.max()
                if highest > 0 and lowest / highest < 0.2:
                    issues.append(f"Group imbalance detected in '{col}': one group is underrepresented.")
            recommendations.append(f"Review and balance '{col}' across groups.")

        risk_level = "Medium"
        if any("underrepresented" in issue.lower() for issue in issues) or any("missing" in issue.lower() for issue in issues):
            risk_level = "High"

        return {
            "bias_risk": risk_level,
            "sensitive_features": detected,
            "issues": issues,
            "recommendations": recommendations,
        }

    def evaluate_model_fairness(
        self,
        predictions: List[Any],
        actual_values: List[Any],
        sensitive_groups: Dict[str, List[Any]],
    ) -> Dict[str, Any]:
        if not sensitive_groups:
            return {
                "fairness_score": 50.0,
                "risk_level": "Unknown",
                "group_metrics": {},
                "warnings": ["No sensitive groups supplied for fairness evaluation."],
            }

        group_metrics: Dict[str, Dict[str, Any]] = {}
        warnings_list: List[str] = []
        base_accuracy = self._accuracy(predictions, actual_values)
        disparity_score = 0.0

        for group_name, indices in sensitive_groups.items():
            group_preds = [predictions[i] for i in indices if i < len(predictions)]
            group_actual = [actual_values[i] for i in indices if i < len(actual_values)]
            if not group_preds or not group_actual:
                warnings_list.append(f"Group {group_name} has no valid samples.")
                continue
            group_acc = self._accuracy(group_preds, group_actual)
            group_fp = self._false_positive_rate(group_preds, group_actual)
            group_fn = self._false_negative_rate(group_preds, group_actual)
            group_metrics[group_name] = {
                "accuracy": group_acc,
                "false_positive_rate": group_fp,
                "false_negative_rate": group_fn,
                "sample_count": len(group_preds),
            }
            disparity_score += abs(base_accuracy - group_acc)
            disparity_score += group_fp + group_fn

        fairness_score = max(0, 100 - min(100, disparity_score * 50))
        risk_level = "Low" if fairness_score >= 80 else "Medium" if fairness_score >= 60 else "High"

        return {
            "fairness_score": round(fairness_score, 1),
            "risk_level": risk_level,
            "group_metrics": group_metrics,
            "warnings": warnings_list,
        }

    def analyze_privacy_risk(self, df: pd.DataFrame) -> Dict[str, Any]:
        detected = []
        recommendations: List[str] = []

        for col in df.columns:
            lower = col.lower()
            for keyword in self.PRIVACY_KEYS:
                if keyword in lower:
                    detected.append(col)
                    break

        if detected:
            recommendations.append("Remove or anonymize direct identifiers before sharing data.")
            recommendations.append("Mask sensitive columns and minimize personally identifiable information.")
            privacy_risk = "High"
        else:
            recommendations.append("Privacy risk appears low based on column names.")
            privacy_risk = "Low"

        unique_identifiers = [col for col in df.columns if df[col].nunique(dropna=False) == len(df)]
        if unique_identifiers:
            detected.extend([col for col in unique_identifiers if col not in detected])
            recommendations.append("Check unique identifiers and remove them if they are not required.")
            if privacy_risk != "High":
                privacy_risk = "Medium"

        if any(col.lower() in self.PRIVACY_KEYS for col in df.columns):
            privacy_risk = "High"

        return {
            "privacy_risk": privacy_risk,
            "detected_identifiers": detected,
            "recommendations": recommendations,
        }

    def generate_ethics_report(
        self,
        bias_report: Dict[str, Any],
        fairness_report: Dict[str, Any],
        privacy_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        risk_levels = [bias_report.get("bias_risk", "Low"), fairness_report.get("risk_level", "Low"), privacy_report.get("privacy_risk", "Low")]
        if "High" in risk_levels:
            overall = "High"
        elif "Medium" in risk_levels:
            overall = "Medium"
        else:
            overall = "Low"

        return {
            "executive_summary": "The ethics assessment identifies bias, fairness, and privacy risks in the dataset and model predictions.",
            "bias_concerns": bias_report.get("issues", []),
            "fairness_concerns": [f"{group}: accuracy {metrics.get('accuracy')}" for group, metrics in fairness_report.get("group_metrics", {}).items()],
            "privacy_concerns": privacy_report.get("detected_identifiers", []),
            "compliance_recommendations": self.recommend_mitigation({
                "bias": bias_report,
                "fairness": fairness_report,
                "privacy": privacy_report,
            }),
            "risk_level": overall,
        }

    def recommend_mitigation(self, risk_report: Dict[str, Any]) -> Dict[str, List[str]]:
        bias = risk_report.get("bias", {})
        fairness = risk_report.get("fairness", {})
        privacy = risk_report.get("privacy", {})

        bias_actions: List[str] = []
        fairness_actions: List[str] = []
        privacy_actions: List[str] = []

        if bias.get("sensitive_features"):
            bias_actions.extend([
                "Collect balanced samples across sensitive groups.",
                "Use stratified sampling for training and validation.",
                "Reweight classes or resample underrepresented groups.",
            ])

        if fairness.get("fairness_score") is not None and isinstance(fairness.get("fairness_score"), (int, float)) and fairness.get("fairness_score") < 80:
            fairness_actions.extend([
                "Evaluate groups separately and measure disparities.",
                "Adjust decision thresholds per group.",
                "Improve data diversity and group representation.",
            ])

        if privacy.get("detected_identifiers"):
            privacy_actions.extend([
                "Remove direct identifiers from the dataset.",
                "Mask or hash sensitive personal data.",
                "Apply anonymization or pseudonymization before sharing.",
            ])

        if not bias_actions:
            bias_actions.append("Continue regular bias audits and documentation.")
        if not fairness_actions:
            fairness_actions.append("Monitor fairness metrics during model updates.")
        if not privacy_actions:
            privacy_actions.append("Maintain minimal data collection and retention practices.")

        return {
            "bias": bias_actions,
            "fairness": fairness_actions,
            "privacy": privacy_actions,
        }

    def calculate_ai_governance_score(
        self,
        bias_report: Dict[str, Any],
        fairness_report: Dict[str, Any],
        privacy_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        score = 100
        if bias_report.get("bias_risk") == "Medium":
            score -= 20
        elif bias_report.get("bias_risk") == "High":
            score -= 40

        if fairness_report.get("risk_level") == "Medium":
            score -= 20
        elif fairness_report.get("risk_level") == "High":
            score -= 40

        if privacy_report.get("privacy_risk") == "Medium":
            score -= 20
        elif privacy_report.get("privacy_risk") == "High":
            score -= 40

        score = max(0, min(100, score))
        grade = "A" if score >= 80 else "B" if score >= 60 else "C" if score >= 40 else "D"
        readiness = "Production Ready" if score >= 80 else "Needs Review"

        return {
            "score": score,
            "grade": grade,
            "readiness": readiness,
        }

    def _accuracy(self, preds: List[Any], actual: List[Any]) -> float:
        if not preds or not actual or len(preds) != len(actual):
            return 0.0
        correct = sum(1 for p, a in zip(preds, actual) if p == a)
        return correct / len(preds)

    def _false_positive_rate(self, preds: List[Any], actual: List[Any]) -> float:
        positives = 0
        false_positives = 0
        for p, a in zip(preds, actual):
            if p == 1:
                positives += 1
                if a != 1:
                    false_positives += 1
        return false_positives / positives if positives else 0.0

    def _false_negative_rate(self, preds: List[Any], actual: List[Any]) -> float:
        negatives = 0
        false_negatives = 0
        for p, a in zip(preds, actual):
            if a == 1:
                negatives += 1
                if p != 1:
                    false_negatives += 1
        return false_negatives / negatives if negatives else 0.0

"""Dataset Intelligence Agent for AutoDS Agent.

Provides advanced AI dataset analysis that behaves like a senior data scientist.
Detects domains, identifies ML problems, assesses risks, recommends models,
and generates project roadmaps.

This module only reads DataFrames and does not modify application state.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re
import pandas as pd
import numpy as np


class DatasetIntelligenceAgent:
    """AI agent that analyzes datasets like a senior data scientist."""

    # Known dataset patterns for accurate recognition
    KNOWN_DATASETS = {
        "Iris": {
            "keywords": ["iris", "sepal", "petal", "species"],
            "domain": "Botanical Science",
            "problem_type": "Multi-Class Classification",
            "target": "species",
        },
        "Heart Disease": {
            "keywords": ["heart", "disease", "cholesterol", "chest_pain", "blood_pressure", "mean radius", "worst radius", "concavity", "target"],
            "domain": "Healthcare",
            "problem_type": "Binary Classification",
            "target": "target",
        },
        "Titanic": {
            "keywords": ["titanic", "survived", "passenger", "cabin", "fare"],
            "domain": "Transportation / Survival Analysis",
            "problem_type": "Binary Classification",
            "target": "survived",
        },
        "Housing": {
            "keywords": ["housing", "price", "area", "bedrooms", "bathrooms"],
            "domain": "Real Estate",
            "problem_type": "Regression",
            "target": "price",
        },
        "Wine": {
            "keywords": ["wine", "alcohol", "acidity", "sugar", "quality"],
            "domain": "Food Chemistry",
            "problem_type": "Classification",
            "target": "quality",
        },
        "Telco Churn": {
            "keywords": ["telco", "churn", "tenure", "monthlycharges", "totalcharges", "customer"],
            "domain": "Customer Analytics",
            "problem_type": "Binary Classification",
            "target": "Churn",
        },
    }

    # Target column name patterns
    CLASSIFICATION_TARGET_NAMES = [
        "target", "class", "label", "species", "diagnosis", "disease",
        "outcome", "survived", "churn", "status", "category", "type"
    ]
    REGRESSION_TARGET_NAMES = [
        "price", "sale_price", "salary", "income", "amount", "value",
        "score", "quality"
    ]

    # Domain keyword mappings
    DOMAIN_KEYWORDS = {
        "Healthcare": [
            "patient", "diagnosis", "age", "blood pressure", "cholesterol",
            "heart", "disease", "medical", "hospital", "treatment", "symptom",
            "health", "cancer", "diabetes", "stroke", "pressure", "glucose"
        ],
        "Finance": [
            "loan", "credit", "income", "transaction", "balance", "debt",
            "interest", "bank", "investment", "stock", "profit", "revenue",
            "price", "amount", "payment", "default"
        ],
        "Real Estate": [
            "price", "area", "bedrooms", "bathrooms", "location", "house",
            "property", "rent", "square", "floor", "zone", "neighborhood",
            "apartment", "building"
        ],
        "Customer Analytics": [
            "customer", "churn", "subscription", "purchase", "complaint",
            "satisfaction", "retention", "segment", "profile", "behavior",
            "loyalty", "engagement"
        ],
        "Marketing/Sales": [
            "sales", "revenue", "campaign", "conversion", "click", "impression",
            "engagement", "marketing", "ad", "promotion", "discount", "roi"
        ],
        "Retail/E-commerce": [
            "product", "order", "quantity", "cart", "checkout", "category",
            "inventory", "supplier", "warehouse", "shipping"
        ],
        "Sports/Gaming": [
            "player", "team", "score", "game", "match", "win", "loss",
            "season", "performance", "rank", "league"
        ],
        "Social Media": [
            "user", "post", "like", "comment", "share", "follower",
            "engagement", "hashtag", "tweet", "sentiment"
        ],
    }

    def __init__(self, df: pd.DataFrame, name: str = "Untitled Dataset") -> None:
        """Initialize the agent with a DataFrame."""
        self.df = df
        self.name = name
        self.columns = list(df.columns)
        self.dtypes = df.dtypes
        self.shape = df.shape

    def _column_matches_target_names(self, col: str, names: List[str]) -> bool:
        """Match whole target tokens (avoids 'status' matching 'furnishingstatus')."""
        col_lower = col.lower()
        tokens = [token for token in re.split(r"[^a-z0-9]+", col_lower) if token]
        for name in names:
            name_lower = name.lower()
            if col_lower == name_lower or name_lower in tokens:
                return True
        return False

    def _find_best_target_column(self) -> Optional[str]:
        """Intelligently find the most likely target column."""
        known = self._detect_known_dataset()
        if known:
            hint = known.get("target")
            if hint:
                for col in self.columns:
                    if col.lower() == str(hint).lower():
                        return col
                if hint in self.columns:
                    return hint

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=["object"]).columns.tolist()

        # Check for explicit target name patterns
        for col in self.columns:
            if self._column_matches_target_names(col, ["target", "label"]):
                return col

        # Regression targets before generic classification names like "status"
        for col in self.columns:
            if self._column_matches_target_names(col, self.REGRESSION_TARGET_NAMES):
                return col

        # Classification-specific target names
        for col in self.columns:
            if self._column_matches_target_names(col, self.CLASSIFICATION_TARGET_NAMES):
                return col

        # Default to last column if it's numeric or categorical
        if self.columns:
            last_col = self.columns[-1]
            if last_col in numeric_cols or last_col in categorical_cols:
                return last_col

        return None

    def _detect_known_dataset(self) -> Optional[Dict[str, Any]]:
        """Check if this is a known dataset and return known information."""
        name_lower = self.name.lower()
        cols_lower = [c.lower() for c in self.columns]

        for dataset_name, info in self.KNOWN_DATASETS.items():
            keywords = info["keywords"]
            matched = sum(1 for kw in keywords if any(kw in col or kw in name_lower for col in cols_lower))
            # If at least 40% of keywords match, it's a known dataset
            if matched >= max(1, len(keywords) * 0.4):
                confidence = round(min(matched / max(len(keywords), 1), 1.0), 2)
                return {
                    "is_known": True,
                    "name": dataset_name,
                    "display_name": info.get("display_name") or dataset_name,
                    "domain": info["domain"],
                    "problem_type": info["problem_type"],
                    "target_hint": info["target"],
                    "confidence": confidence,
                }

        return None

    def detect_domain(self) -> Dict[str, Any]:
        """Detect the domain/industry of the dataset."""
        # Check for known datasets first
        known = self._detect_known_dataset()
        if known:
            return {
                "domain": known["domain"],
                "confidence": 0.95,
                "reason": f"Recognized as {known['name']} dataset based on column names and filename.",
            }

        name_lower = self.name.lower()
        cols_lower = [c.lower() for c in self.columns]
        all_text = " ".join([name_lower] + cols_lower)

        domain_scores: Dict[str, float] = {d: 0.0 for d in self.DOMAIN_KEYWORDS}

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in all_text:
                    domain_scores[domain] += 1.0

        if not domain_scores or max(domain_scores.values()) == 0:
            return {
                "domain": "General Machine Learning Dataset",
                "confidence": 0.0,
                "reason": "No specific domain keywords detected.",
            }

        best_domain = max(domain_scores, key=domain_scores.get)
        best_score = domain_scores[best_domain]
        max_possible = len(self.DOMAIN_KEYWORDS[best_domain])
        confidence = min(best_score / max(max_possible, 1), 1.0)

        reason = f"Detected keywords related to {best_domain} in dataset name and columns."

        return {
            "domain": best_domain,
            "confidence": float(round(confidence, 2)),
            "reason": reason,
            "all_scores": {d: float(round(v, 2)) for d, v in domain_scores.items()},
        }

    def analyze_problem(self) -> Dict[str, Any]:
        """Detect the machine learning problem type."""
        # Check for known datasets first
        known = self._detect_known_dataset()
        if known:
            target_hint = known.get("target")
            target_col = None
            if target_hint:
                for col in self.columns:
                    if col.lower() == str(target_hint).lower():
                        target_col = col
                        break
                if target_col is None and target_hint in self.columns:
                    target_col = target_hint
            if target_col is None:
                target_col = self._find_best_target_column() or target_hint or "target"
            if known["problem_type"] == "Binary Classification":
                recommended_metrics = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
            elif known["problem_type"] == "Multi-Class Classification":
                recommended_metrics = ["Precision", "Recall", "F1-Score", "Macro-F1"]
            else:  # Regression
                recommended_metrics = ["MAE", "RMSE", "R²", "MAPE"]

            return {
                "problem_type": known["problem_type"],
                "likely_target": target_col,
                "recommended_metrics": recommended_metrics,
                "reason": f"Recognized dataset: {known['problem_type']} problem.",
            }

        # Find target column intelligently
        target_col = self._find_best_target_column()
        if not target_col:
            return {
                "problem_type": "Unknown",
                "likely_target": None,
                "reason": "Unable to identify target column.",
            }

        target_data = self.df[target_col].dropna()
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=["object"]).columns.tolist()

        # Check if target is categorical
        is_categorical = target_col in categorical_cols or self.df[target_col].dtype == "object"

        if is_categorical:
            unique_count = target_data.nunique()
            if unique_count == 2:
                problem_type = "Binary Classification"
                recommended_metrics = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
                reason = "Binary target detected (two distinct classes)."
            elif 2 < unique_count <= 20:
                problem_type = "Multi-Class Classification"
                recommended_metrics = ["Precision", "Recall", "F1-Score", "Macro-F1"]
                reason = f"Categorical target with {unique_count} classes detected."
            else:
                problem_type = "Regression (Categorical Encoded)"
                recommended_metrics = ["MAE", "RMSE", "R²"]
                reason = "High cardinality categorical target; treat as regression."
        else:
            # Numeric target - check for classification or regression
            unique_count = target_data.nunique()
            if unique_count <= 20 and unique_count > 1:
                # Could be encoded classification
                problem_type = "Classification (Likely Binary/Multi-Class)"
                recommended_metrics = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
                reason = f"Numeric target with {unique_count} distinct values detected; may be encoded classes."
            else:
                problem_type = "Regression"
                recommended_metrics = ["MAE", "RMSE", "R²", "MAPE"]
                reason = "Continuous numerical target detected."

        return {
            "problem_type": problem_type,
            "likely_target": target_col,
            "recommended_metrics": recommended_metrics,
            "reason": reason,
        }

    def assess_risks(self) -> Dict[str, Any]:
        """Assess data quality and ML risks."""
        risks: List[str] = []
        risk_scores: Dict[str, float] = {}

        # Missing values
        missing_pct = (self.df.isnull().sum().sum()) / (self.shape[0] * self.shape[1]) * 100
        if missing_pct > 30:
            risks.append(f"High missing data ({missing_pct:.1f}%). Consider imputation or removal.")
            risk_scores["missing_values"] = 3.0
        elif missing_pct > 10:
            risks.append(f"Moderate missing data ({missing_pct:.1f}%). Review and address.")
            risk_scores["missing_values"] = 2.0
        elif missing_pct > 0:
            risks.append(f"Low missing data ({missing_pct:.1f}%). Plan imputation strategy.")
            risk_scores["missing_values"] = 1.0
        else:
            risk_scores["missing_values"] = 0.0

        # Duplicates
        dup_pct = (self.df.duplicated().sum()) / self.shape[0] * 100
        if dup_pct > 10:
            risks.append(f"Significant duplicates ({dup_pct:.1f}%). Remove before modeling.")
            risk_scores["duplicates"] = 3.0
        elif dup_pct > 1:
            risks.append(f"Minor duplicates ({dup_pct:.1f}%). Consider removal.")
            risk_scores["duplicates"] = 1.0
        else:
            risk_scores["duplicates"] = 0.0

        # Dataset size
        if self.shape[0] < 100:
            risks.append("Very small dataset. Risk of overfitting. Collect more data if possible.")
            risk_scores["small_dataset"] = 3.0
        elif self.shape[0] < 500:
            risks.append("Dataset is small. Use cross-validation and regularization carefully.")
            risk_scores["small_dataset"] = 2.0
        else:
            risk_scores["small_dataset"] = 0.0

        # High dimensionality
        ratio = self.shape[1] / max(self.shape[0], 1)
        if ratio > 0.5:
            risks.append(f"High dimensionality ({self.shape[1]} features for {self.shape[0]} rows). Use feature selection.")
            risk_scores["high_dimensionality"] = 3.0
        elif ratio > 0.2:
            risks.append(f"Moderate dimensionality. Consider feature engineering.")
            risk_scores["high_dimensionality"] = 1.0
        else:
            risk_scores["high_dimensionality"] = 0.0

        # Class imbalance (if classification)
        problem = self.analyze_problem()
        if "Classification" in problem.get("problem_type", ""):
            target = problem.get("likely_target")
            if target:
                target_counts = self.df[target].value_counts(normalize=True)
                if len(target_counts) >= 2:
                    imbalance_ratio = target_counts.iloc[0] / target_counts.iloc[-1]
                    if imbalance_ratio > 10:
                        risks.append(f"Severe class imbalance ({imbalance_ratio:.1f}:1). Use stratified sampling and class weights.")
                        risk_scores["class_imbalance"] = 3.0
                    elif imbalance_ratio > 3:
                        risks.append(f"Moderate class imbalance ({imbalance_ratio:.1f}:1). Consider SMOTE or class weights.")
                        risk_scores["class_imbalance"] = 2.0
                    else:
                        risk_scores["class_imbalance"] = 0.0

        # Data leakage detection
        suspicious_cols = [c for c in self.columns if any(
            word in c.lower() for word in [
                "outcome_copy", "prediction", "predicted", "result_copy", "target_copy",
                "final_result", "target_encoded", "y_train", "future_value"
            ]
        )]
        if suspicious_cols:
            risks.append(f"Potential data leakage detected in columns: {', '.join(suspicious_cols)}. Review before modeling.")
            risk_scores["data_leakage"] = 3.0
        else:
            risk_scores["data_leakage"] = 0.0

        # Very small dataset risk
        if self.shape[0] < 80:
            risks.append(f"Critical: Very small dataset ({self.shape[0]} rows). High overfitting risk.")
            risk_scores["very_small_dataset"] = 3.0
        else:
            risk_scores["very_small_dataset"] = 0.0

        # High dimensional risk
        if self.shape[1] > self.shape[0]:
            risks.append(f"Critical: More columns ({self.shape[1]}) than rows ({self.shape[0]}). Curse of dimensionality.")
            risk_scores["high_dimensional"] = 3.0
        else:
            risk_scores["high_dimensional"] = 0.0

        # Overall risk level
        total_risk = sum(risk_scores.values())
        if total_risk >= 8:
            risk_level = "High"
        elif total_risk >= 4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            "risk_level": risk_level,
            "risk_score": float(round(total_risk, 1)),
            "risks": risks,
            "detailed_scores": risk_scores,
        }

    def recommend_models(self) -> Dict[str, Any]:
        """Recommend machine learning models based on problem type and domain."""
        problem = self.analyze_problem()
        problem_type = problem.get("problem_type", "Unknown")
        domain = self.detect_domain()
        domain_name = domain.get("domain", "General")

        recommendations = []

        if "Binary Classification" in problem_type:
            if domain_name == "Healthcare":
                recommendations = [
                    {"model": "Logistic Regression", "reason": "Interpretable baseline, medically acceptable."},
                    {"model": "Random Forest", "reason": "Captures complex interactions, provides feature importance."},
                    {"model": "XGBoost", "reason": "High accuracy with built-in interpretability."},
                ]
            elif domain_name == "Finance":
                recommendations = [
                    {"model": "Logistic Regression", "reason": "Industry standard for risk modeling."},
                    {"model": "Gradient Boosting", "reason": "Superior to Logistic Regression, interpretable."},
                    {"model": "SVM", "reason": "Excellent for binary classification with clear boundaries."},
                ]
            elif domain_name == "Transportation / Survival Analysis":
                recommendations = [
                    {"model": "Logistic Regression", "reason": "Classic survival prediction baseline."},
                    {"model": "Random Forest", "reason": "Captures passenger interaction patterns."},
                    {"model": "XGBoost", "reason": "Best-in-class for survival prediction."},
                ]
            else:
                recommendations = [
                    {"model": "Logistic Regression", "reason": "Simple interpretable baseline."},
                    {"model": "Random Forest", "reason": "Robust to outliers, feature importance."},
                    {"model": "XGBoost", "reason": "State-of-the-art binary classification."},
                ]

        elif "Multi-Class Classification" in problem_type:
            if domain_name == "Botanical Science":
                recommendations = [
                    {"model": "Logistic Regression", "reason": "Effective multiclass baseline."},
                    {"model": "Random Forest", "reason": "Naturally handles multiclass, great for botanical data."},
                    {"model": "SVM", "reason": "High performance on multiclass problems."},
                ]
            else:
                recommendations = [
                    {"model": "Random Forest", "reason": "Handles multiclass naturally, fast."},
                    {"model": "XGBoost", "reason": "Superior multiclass with optimized loss."},
                    {"model": "Neural Network", "reason": "Deep learning for complex patterns."},
                ]

        elif "Regression" in problem_type:
            if domain_name == "Real Estate":
                recommendations = [
                    {"model": "Linear Regression", "reason": "Baseline for property valuation."},
                    {"model": "Random Forest Regressor", "reason": "Non-linear, captures location effects."},
                    {"model": "Gradient Boosting Regressor", "reason": "Best for real estate price prediction."},
                ]
            elif domain_name == "Food Chemistry":
                recommendations = [
                    {"model": "Random Forest", "reason": "Captures complex chemical interactions."},
                    {"model": "Gradient Boosting", "reason": "Superior performance on chemistry data."},
                    {"model": "SVM", "reason": "Good at finding non-linear relationships."},
                ]
            else:
                recommendations = [
                    {"model": "Linear Regression", "reason": "Interpretable baseline."},
                    {"model": "Random Forest Regressor", "reason": "Handles non-linear relationships."},
                    {"model": "XGBoost Regressor", "reason": "Best overall regression performance."},
                ]

        else:
            recommendations = [
                {"model": "Random Forest", "reason": "Versatile, works for most problems."},
                {"model": "XGBoost", "reason": "High performance, feature importance."},
                {"model": "Neural Network", "reason": "Deep learning for complex patterns."},
            ]

        return {
            "problem_type": problem_type,
            "recommendations": recommendations,
            "metric_focus": problem.get("recommended_metrics", []),
        }

    def generate_roadmap(self) -> List[str]:
        """Generate a step-by-step project roadmap."""
        problem = self.analyze_problem()
        domain = self.detect_domain()
        risks = self.assess_risks()

        roadmap = []

        # Step 1: Data quality
        if risks.get("risks"):
            roadmap.append(f"Step 1: Address data quality issues ({len(risks['risks'])} issues identified). Impute missing values, remove duplicates.")
        else:
            roadmap.append("Step 1: Data quality is good. Proceed with exploratory analysis.")

        # Step 2: EDA
        domain_name = domain.get("domain", "General")
        if domain_name != "General Purpose":
            roadmap.append(f"Step 2: Analyze patterns specific to {domain_name} domain. Focus on domain-relevant features.")
        else:
            roadmap.append("Step 2: Perform exploratory data analysis. Discover relationships and distributions.")

        # Step 3: Feature engineering
        if self.shape[1] > 50:
            roadmap.append("Step 3: High dimensionality detected. Perform feature selection and dimensionality reduction.")
        else:
            roadmap.append("Step 3: Create new features based on domain knowledge and EDA insights.")

        # Step 4: Model training
        problem_type = problem.get("problem_type", "Unknown")
        roadmap.append(f"Step 4: Train multiple {problem_type.lower()} models using AutoML Studio.")

        # Step 5: Model optimization
        if "Classification" in problem_type:
            roadmap.append("Step 5: Optimize for Recall/F1-Score and other domain-relevant metrics. Use cross-validation.")
        else:
            roadmap.append("Step 5: Optimize for MAE/RMSE. Fine-tune hyperparameters.")

        # Step 6: Explainability
        roadmap.append("Step 6: Generate SHAP explanations in AI Decision Intelligence. Understand feature impacts.")

        # Step 7: Validation and deployment
        if domain_name == "Healthcare":
            roadmap.append("Step 7: Validate model with domain experts. Ensure clinical acceptability before deployment.")
        else:
            roadmap.append("Step 7: Validate model on held-out test set. Deploy with monitoring.")

        roadmap.append("Step 8: Generate AI Research Report with findings and recommendations.")

        return roadmap

    def calculate_intelligence_score(self) -> Dict[str, Any]:
        """Calculate overall dataset intelligence score (0-100)."""
        score = 75.0  # Start with baseline

        # Data quality component (max +25)
        missing_pct = (self.df.isnull().sum().sum()) / (self.shape[0] * self.shape[1]) * 100
        quality_penalty = min(missing_pct / 10, 15)
        score -= quality_penalty

        dup_pct = (self.df.duplicated().sum()) / self.shape[0] * 100
        dup_penalty = min(dup_pct / 5, 10)
        score -= dup_penalty

        # Dataset size component (max ±15)
        if self.shape[0] < 100:
            score -= 15
        elif self.shape[0] < 500:
            score -= 10
        elif self.shape[0] > 100000:
            score += 5

        # Balance component (max ±10)
        problem = self.analyze_problem()
        if "Classification" in problem.get("problem_type", ""):
            target = problem.get("likely_target")
            if target:
                target_counts = self.df[target].value_counts(normalize=True)
                if len(target_counts) >= 2:
                    imbalance_ratio = target_counts.iloc[0] / target_counts.iloc[-1]
                    if imbalance_ratio > 10:
                        score -= 10
                    elif imbalance_ratio > 3:
                        score -= 5

        # Dimensionality component (max ±10)
        ratio = self.shape[1] / max(self.shape[0], 1)
        if ratio > 0.5:
            score -= 10
        elif ratio > 0.2:
            score -= 5

        # Risk component
        risks = self.assess_risks()
        if risks.get("risk_level") == "High":
            score -= 15
        elif risks.get("risk_level") == "Medium":
            score -= 5

        # Domain detection bonus
        domain = self.detect_domain()
        if domain.get("confidence", 0) > 0.5:
            score += 5

        score = max(0.0, min(100.0, score))

        if score < 40:
            label = "High Risk Dataset"
        elif score < 70:
            label = "Needs Attention"
        elif score < 90:
            label = "Good Dataset"
        else:
            label = "Excellent Dataset"

        return {
            "score": float(round(score, 1)),
            "label": label,
            "components": {
                "quality": 25 - quality_penalty - dup_penalty,
                "size": 15 - (15 if self.shape[0] < 100 else 10 if self.shape[0] < 500 else -5),
                "balance": 10 - (10 if "Classification" in problem.get("problem_type", "") else 0),
                "dimensionality": 10 - (10 if ratio > 0.5 else 5 if ratio > 0.2 else 0),
            },
        }

    def generate_dataset_report(self) -> Dict[str, Any]:
        """Generate a comprehensive dataset intelligence report."""
        known = self._detect_known_dataset()
        domain = self.detect_domain()
        problem = self.analyze_problem()
        risks = self.assess_risks()
        models = self.recommend_models()
        roadmap = self.generate_roadmap()
        score = self.calculate_intelligence_score()

        # Generate professional executive summary
        domain_name = domain.get("domain", "General")
        problem_type = problem.get("problem_type", "Unknown")
        risk_level = risks.get("risk_level", "Unknown")
        top_model = models["recommendations"][0]["model"] if models["recommendations"] else "Unknown"
        metrics = ", ".join(models["metric_focus"][:2]) if models["metric_focus"] else "Accuracy"

        # Build a professional summary
        executive_summary = (
            f"This {domain_name} dataset represents a {problem_type} problem. "
            f"Dataset contains {self.shape[0]:,} records and {self.shape[1]} features with {risk_level.lower()} risk. "
            f"Intelligence Score: {score['score']}/100 ({score['label']}). "
            f"Data Quality: {score['components'].get('quality', 0):.1f}/25 points. "
            f"Recommended Approach: Start with {top_model}, optimize for {metrics}. "
            f"Priority: {risk_level} risk dataset requires careful data preparation before modeling."
        )

        return {
            "dataset_name": self.name,
            "dataset_shape": {"rows": self.shape[0], "columns": self.shape[1]},
            "dataset_identification": {
                "uploaded_file": self.name,
                "detected_dataset": (
                    known.get("display_name") or known.get("name")
                    if known and known.get("confidence", 0) >= 0.5
                    else None
                ),
                "detection_confidence": float(known.get("confidence", 0)) if known else 0.0,
            },
            "domain_analysis": domain,
            "problem_analysis": problem,
            "risk_analysis": risks,
            "model_strategy": models,
            "roadmap": roadmap,
            "intelligence_score": score,
            "executive_summary": executive_summary,
        }

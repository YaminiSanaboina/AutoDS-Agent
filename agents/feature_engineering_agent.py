"""Feature engineering recommendations for AutoDS Agent.

This backend module analyzes dataset columns and returns intelligent feature
transformation recommendations without modifying UI pages.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class FeatureEngineeringAgent:
    """Agent that generates feature engineering insights for AutoDS."""

    def __init__(self, df: pd.DataFrame, name: str = "Untitled Dataset") -> None:
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.name = name
        self.columns = list(self.df.columns)

    def analyze_features(self) -> Dict[str, Dict[str, Any]]:
        """Analyze each feature and return its type, missing rate, cardinality, and recommendation."""
        report: Dict[str, Dict[str, Any]] = {}
        for col in self.columns:
            series = self.df[col]
            dtype = series.dtype
            missing_percent = float(series.isna().mean() * 100)
            unique_values = int(series.nunique(dropna=True))
            feature_type = self._detect_feature_type(series)
            recommendation = self._recommend_by_type(col, series, feature_type)
            report[col] = {
                "type": feature_type,
                "missing_percent": round(missing_percent, 2),
                "unique_values": unique_values,
                "recommendation": recommendation,
            }
        return report

    def _detect_feature_type(self, series: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(series):
            return "Boolean"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "Datetime"
        if pd.api.types.is_numeric_dtype(series):
            if self._likely_identifier(series):
                return "Identifier"
            return "Numerical"
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            unique_values = series.nunique(dropna=True)
            unique_ratio = unique_values / max(1, len(series))
            avg_len = series.dropna().astype(str).str.len().mean() if not series.dropna().empty else 0
            if unique_ratio > 0.9 and unique_values:
                return "Identifier"
            if avg_len > 20 or unique_values > 50:
                return "Text"
            return "Categorical"
        return "Categorical"

    def _likely_identifier(self, series: pd.Series) -> bool:
        if pd.api.types.is_integer_dtype(series) or pd.api.types.is_string_dtype(series):
            unique_count = series.nunique(dropna=True)
            if unique_count >= len(series) * 0.9 and unique_count > 5:
                return True
        return False

    def _recommend_by_type(self, col: str, series: pd.Series, feature_type: str) -> str:
        if feature_type == "Numerical":
            if series.nunique(dropna=True) <= 10:
                return "Small numerical feature. Consider binning or polynomial features."
            return "Apply scaling or normalization before training."
        if feature_type == "Categorical":
            return "Encode this feature before modeling; use one-hot or target encoding based on cardinality."
        if feature_type == "Boolean":
            return "Convert True/False values to 1/0 for modeling."
        if feature_type == "Datetime":
            return "Extract date parts like year, month, day, and weekday."
        if feature_type == "Text":
            return "Create text embeddings, token counts, or simple length-based features."
        if feature_type == "Identifier":
            return "Exclude identifier features from training unless they provide business meaning."
        return "Review the feature and choose a suitable transformation."

    def recommend_missing_value_strategy(self) -> Dict[str, str]:
        """Recommend a missing value strategy for each feature."""
        strategies: Dict[str, str] = {}
        for col in self.columns:
            series = self.df[col]
            feature_type = self._detect_feature_type(series)
            if series.isna().sum() == 0:
                continue
            if feature_type == "Numerical":
                strategies[col] = "Fill missing values using median"
            elif feature_type == "Categorical" or feature_type == "Boolean" or feature_type == "Identifier":
                strategies[col] = "Fill missing values using mode"
            elif feature_type == "Datetime":
                strategies[col] = "Fill missing values using forward fill or domain-specific handling"
            else:
                strategies[col] = "Handle missing values with a strategy appropriate for the feature type"
        return strategies

    def recommend_encoding(self) -> Dict[str, str]:
        """Recommend encoding strategies for categorical and boolean features."""
        encoding: Dict[str, str] = {}
        for col in self.columns:
            series = self.df[col]
            feature_type = self._detect_feature_type(series)
            if feature_type == "Boolean":
                encoding[col] = "Convert True/False to 0/1"
            elif feature_type == "Categorical":
                cardinality = series.nunique(dropna=True)
                if cardinality <= 10:
                    encoding[col] = "One-Hot Encoding"
                else:
                    encoding[col] = "Target encoding or frequency encoding"
            elif feature_type == "Text":
                encoding[col] = "Use text vectorization or embeddings"
        return encoding

    def suggest_new_features(self) -> Dict[str, str]:
        """Suggest useful new feature creations based on dataset characteristics."""
        suggestions: Dict[str, str] = {}
        detected = self.analyze_features()
        numeric_cols = [c for c, meta in detected.items() if meta["type"] == "Numerical"]
        datetime_cols = [c for c, meta in detected.items() if meta["type"] == "Datetime"]
        categorical_cols = [c for c, meta in detected.items() if meta["type"] == "Categorical"]

        if "area" in self.columns and "price" in self.columns:
            suggestions["price_per_area"] = "Create price per area feature for better housing value modeling."
        if "bedrooms" in self.columns and "area" in self.columns:
            suggestions["room_density"] = "Create room density by dividing bedrooms by area."
        if "sibsp" in self.columns or "parch" in self.columns:
            if "sibsp" in self.columns and "parch" in self.columns:
                suggestions["family_size"] = "Create family size from siblings/spouses and parents/children."
        if any(keyword in self.name.lower() for keyword in ["titanic"]):
            if "name" in self.columns:
                suggestions["title"] = "Extract passenger title from the name column."
            if "cabin" in self.columns:
                suggestions["cabin_indicator"] = "Create a cabin presence indicator feature."
        if any(keyword in self.name.lower() for keyword in ["heart", "disease", "health", "clinical"]):
            if "age" in self.columns:
                suggestions["age_group"] = "Create age group categories for risk segmentation."
            if "cholesterol" in self.columns or "blood_pressure" in self.columns:
                suggestions["health_risk_score"] = "Combine medical indicators into a risk score."
        if any(keyword in self.name.lower() for keyword in ["customer", "churn", "subscription", "sales", "marketing"]):
            suggestions["customer_segment"] = "Create customer segments based on behavior and demographics."

        for col in datetime_cols:
            suggestions[f"{col}_year"] = "Extract the year from the datetime column."
            suggestions[f"{col}_month"] = "Extract the month from the datetime column."
            suggestions[f"{col}_dayofweek"] = "Extract the day of week from the datetime column."

        if len(numeric_cols) >= 2:
            if "area" in self.columns and "price" in self.columns:
                suggestions["price_per_area"] = "Create a ratio feature to capture unit pricing."
            if len(numeric_cols) >= 3:
                first, second = numeric_cols[0], numeric_cols[1]
                suggestions[f"{first}_to_{second}_ratio"] = f"Create a ratio of {first} to {second}."

        return suggestions

    def recommend_feature_selection(self) -> Dict[str, str]:
        """Recommend features to remove based on selection intelligence."""
        recommendations: Dict[str, str] = {}
        df = self.df

        for col in self.columns:
            series = df[col]
            if series.nunique(dropna=False) <= 1:
                recommendations[col] = "Remove constant column."
                continue
            missing_percent = series.isna().mean() * 100
            if missing_percent > 50:
                recommendations[col] = "Remove or impute column with more than 50% missing values."

        duplicates = self._find_duplicate_columns()
        for col in duplicates:
            recommendations[col] = "Remove duplicate column."

        correlated = self._find_highly_correlated_features()
        for col, pair in correlated.items():
            recommendations[col] = f"Remove or combine highly correlated feature with {pair}."

        return recommendations

    def _find_duplicate_columns(self) -> List[str]:
        duplicates: List[str] = []
        df = self.df.fillna("__MISSING__")
        seen: Dict[Tuple[int, ...], str] = {}
        for col in self.columns:
            values = tuple(df[col].astype(str).tolist())
            if values in seen:
                duplicates.append(col)
            else:
                seen[values] = col
        return duplicates

    def _find_highly_correlated_features(self) -> Dict[str, str]:
        numeric = self.df.select_dtypes(include=[np.number])
        result: Dict[str, str] = {}
        if numeric.shape[1] < 2:
            return result
        corr = numeric.corr().abs()
        for i, col in enumerate(corr.columns):
            for j, other in enumerate(corr.columns):
                if i >= j:
                    continue
                if corr.loc[col, other] > 0.9:
                    result[other] = col
        return result

    def detect_leakage(self) -> Dict[str, Any]:
        """Detect suspicious columns that may lead to data leakage."""
        risk_columns: List[str] = []
        keywords = ["target_copy", "prediction", "future", "outcome_result", "leak", "label"]
        for col in self.columns:
            lower = col.lower()
            if any(keyword in lower for keyword in keywords):
                risk_columns.append(col)

        if not risk_columns:
            return {"risk_level": "Low", "suspicious_columns": []}

        level = "Medium"
        if any("future" in c.lower() or "prediction" in c.lower() or "outcome" in c.lower() for c in risk_columns):
            level = "High"
        return {"risk_level": level, "suspicious_columns": risk_columns}

    def generate_feature_plan(self) -> Dict[str, Any]:
        """Generate a roadmap for feature transformations."""
        analyzed = self.analyze_features()
        current_features = len(self.columns)
        selection = self.recommend_feature_selection()
        encoding = self.recommend_encoding()
        missing = self.recommend_missing_value_strategy()
        new_features = self.suggest_new_features()

        recommended_changes = len(selection) + len(encoding) + len(missing) + len(new_features)
        steps: List[str] = [
            "Remove constant and duplicate columns",
            "Encode categorical variables",
            "Impute missing values",
            "Create derived features for dates and ratios",
        ]
        if any(meta["type"] == "Numerical" for meta in analyzed.values()):
            steps.append("Scale or normalize numerical features")
        if any(meta["type"] == "Text" for meta in analyzed.values()):
            steps.append("Apply text feature extraction or embeddings")

        return {
            "current_features": current_features,
            "recommended_changes": recommended_changes,
            "expected_benefit": "Improved model accuracy and stability",
            "steps": steps,
        }

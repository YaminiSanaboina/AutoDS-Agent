from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class SyntheticDataAgent:
    """Agent for analyzing, generating, and evaluating synthetic datasets."""

    TARGET_COLUMN_HINTS = [
        "target",
        "label",
        "class",
        "outcome",
        "status",
        "category",
        "type",
        "y",
        "survived",
        "quality",
        "score",
        "churn",
    ]

    PRIVACY_SENSITIVE_KEYS = [
        "email",
        "phone",
        "ssn",
        "passport",
        "credit_card",
        "address",
        "name",
        "id",
        "account",
    ]

    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        columns = list(df.columns)
        shape = list(df.shape)
        feature_types = {col: self._detect_feature_type(df[col]) for col in columns}

        missing_info = {
            "total_missing": int(df.isna().sum().sum()),
            "missing_by_column": {
                col: int(df[col].isna().sum()) for col in columns if df[col].isna().any()
            },
            "missing_percent": round(float(df.isna().mean().mean() * 100), 2),
        }

        numeric_features = [col for col, ft in feature_types.items() if ft == "Numerical"]
        categorical_features = [col for col, ft in feature_types.items() if ft == "Categorical"]
        class_candidates = self._find_target_candidates(df, feature_types)
        class_imbalance = self._detect_class_imbalance(df, class_candidates)

        recommendation = self.recommend_augmentation_strategy(df)
        synthetic_recommendation = recommendation["recommended_method"]

        return {
            "shape": shape,
            "feature_types": {
                "numerical": numeric_features,
                "categorical": categorical_features,
                "all": feature_types,
                "target_candidates": class_candidates,
            },
            "data_quality": {
                "missing": missing_info,
                "duplicate_rows": int(df.duplicated().sum()),
                "class_imbalance": class_imbalance,
            },
            "synthetic_recommendation": synthetic_recommendation,
        }

    def generate_synthetic_data(
        self,
        df: pd.DataFrame,
        num_samples: int,
        method: str = "statistical",
    ) -> Dict[str, Any]:
        method = method.lower()
        if method not in {"statistical", "bootstrap", "balanced"}:
            method = "statistical"

        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")

        if df.empty:
            synthetic_df = pd.DataFrame()
            quality_score = 0
        elif method == "bootstrap":
            synthetic_df = df.sample(n=num_samples, replace=True, random_state=42).reset_index(drop=True)
            quality_score = self._estimate_quality(df, synthetic_df, method)
        elif method == "balanced":
            synthetic_df = self._generate_balanced(df, num_samples)
            quality_score = self._estimate_quality(df, synthetic_df, method)
        else:
            synthetic_df = self._generate_statistical(df, num_samples)
            quality_score = self._estimate_quality(df, synthetic_df, method)

        return {
            "synthetic_df": synthetic_df,
            "method": method,
            "samples_generated": int(len(synthetic_df)),
            "quality_score": int(round(quality_score)),
        }

    def evaluate_similarity(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        warnings: List[str] = []
        numeric_similarity: Dict[str, Any] = {}
        categorical_similarity: Dict[str, Any] = {}

        common_columns = [col for col in original_df.columns if col in synthetic_df.columns]
        if not common_columns:
            warnings.append("No columns in common between original and synthetic datasets.")

        numeric_cols = [col for col in common_columns if pd.api.types.is_numeric_dtype(original_df[col])]
        cat_cols = [
            col
            for col in common_columns
            if pd.api.types.is_object_dtype(original_df[col])
            or isinstance(original_df[col].dtype, pd.CategoricalDtype)
        ]

        numeric_scores: List[float] = []
        for col in numeric_cols:
            orig = original_df[col].dropna()
            synth = synthetic_df[col].dropna()
            if orig.empty or synth.empty:
                warnings.append(f"Insufficient numeric data for similarity assessment on column '{col}'.")
                continue
            mean_diff = abs(orig.mean() - synth.mean())
            std_diff = abs(orig.std(ddof=0) - synth.std(ddof=0))
            overlap = self._range_overlap(orig, synth)
            score = max(0.0, min(100.0, 100.0 - mean_diff - std_diff + overlap * 20.0))
            numeric_similarity[col] = {
                "mean_diff": round(float(mean_diff), 4),
                "std_diff": round(float(std_diff), 4),
                "range_overlap": round(float(overlap), 4),
                "similarity_score": int(round(score)),
            }
            numeric_scores.append(score)

        categorical_scores: List[float] = []
        for col in cat_cols:
            orig_dist = self._value_distribution(original_df[col])
            synth_dist = self._value_distribution(synthetic_df[col])
            dist_similarity = self._distribution_similarity(orig_dist, synth_dist)
            unseen = [cat for cat in synth_dist.keys() if cat not in orig_dist]
            categorical_similarity[col] = {
                "distribution_similarity": round(float(dist_similarity), 4),
                "unseen_categories": unseen,
            }
            categorical_scores.append(dist_similarity * 100.0)

        score_components: List[float] = []
        if numeric_scores:
            score_components.append(np.mean(numeric_scores))
        if categorical_scores:
            score_components.append(np.mean(categorical_scores))
        similarity_score = int(round(float(np.mean(score_components))) if score_components else 0.0)

        return {
            "similarity_score": similarity_score,
            "numeric_similarity": numeric_similarity,
            "categorical_similarity": categorical_similarity,
            "warnings": warnings,
        }

    def calculate_privacy_score(
        self,
        original_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        issues: List[str] = []
        score = 100

        if synthetic_df.empty or original_df.empty:
            if synthetic_df.empty:
                issues.append("Synthetic dataset is empty.")
            if original_df.empty:
                issues.append("Original dataset is empty.")
            return {
                "privacy_score": 0,
                "risk_level": "High",
                "issues": issues,
            }

        exact_matches = int(
            synthetic_df.merge(original_df).drop_duplicates().shape[0]
        )
        if exact_matches > 0:
            issues.append(f"Found {exact_matches} exact row duplicates from the original dataset.")
            score -= min(40, exact_matches * 5)

        near_duplicates = self._count_near_duplicates(original_df, synthetic_df)
        if near_duplicates > 0:
            issues.append(f"Found {near_duplicates} near-identical synthetic rows.")
            score -= min(30, near_duplicates * 3)

        unique_columns = self._find_unique_identifier_columns(original_df)
        copied_identifiers = []
        for col in unique_columns:
            if col in synthetic_df.columns and synthetic_df[col].isin(original_df[col]).any():
                copied_identifiers.append(col)
        if copied_identifiers:
            issues.append(f"Unique identifiers copied for columns: {', '.join(copied_identifiers)}.")
            score -= 30

        score = max(0, min(100, score))
        if score >= 80:
            risk_level = "Low"
        elif score >= 50:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return {
            "privacy_score": score,
            "risk_level": risk_level,
            "issues": issues,
        }

    def recommend_augmentation_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {
                "recommended_method": "statistical",
                "reasoning": "Empty dataset detected; statistical synthetic generation is the safest way to populate data.",
                "expected_benefits": ["Generate synthetic data without copying exact records.", "Keep feature distributions under control."],
            }

        rows, cols = df.shape
        class_candidates = self._find_target_candidates(df, {col: self._detect_feature_type(df[col]) for col in df.columns})
        sensitive = any(key in col.lower() for col in df.columns for key in self.PRIVACY_SENSITIVE_KEYS)
        imbalance = self._detect_class_imbalance(df, class_candidates)

        if sensitive:
            return {
                "recommended_method": "statistical",
                "reasoning": "Privacy-sensitive columns were detected, so statistical generation reduces the chance of reproducing individual records.",
                "expected_benefits": ["Protects sensitive attributes.", "Generates realistic data while minimizing direct copying."],
            }

        if imbalance.get("is_imbalanced"):
            return {
                "recommended_method": "balanced",
                "reasoning": "Class imbalance exists and balanced generation can improve minority class representation.",
                "expected_benefits": ["Improves model training for underrepresented classes.", "Reduces bias from skewed class proportions."],
            }

        if rows < 500:
            return {
                "recommended_method": "statistical",
                "reasoning": "Small dataset detected, so generating more synthetic samples can increase training stability.",
                "expected_benefits": ["Adds more examples for model training.", "Preserves overall feature behavior."],
            }

        return {
            "recommended_method": "bootstrap",
            "reasoning": "Dataset size is moderate and bootstrapping preserves the original joint distribution without complex modeling.",
            "expected_benefits": ["Maintains realistic relationships between features.", "Simple and efficient generation."],
        }

    def generate_synthetic_data_report(
        self,
        original_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        similarity = self.evaluate_similarity(original_df, synthetic_df)
        privacy = self.calculate_privacy_score(original_df, synthetic_df)
        quality = int(round((similarity["similarity_score"] + privacy["privacy_score"]) / 2))

        benefits = [
            "Enables model development when real data is scarce.",
            "Helps balance imbalanced classes.",
            "Supports privacy-preserving dataset augmentation.",
        ]
        risks: List[str] = []
        if privacy["risk_level"] != "Low":
            risks.append("Risk of data leakage from copied or near-identical records.")
        if similarity["similarity_score"] < 50:
            risks.append("Synthetic data may not reflect the original distribution closely enough.")

        recommendation = self.recommend_augmentation_strategy(original_df)

        return {
            "quality_score": quality,
            "similarity_score": similarity["similarity_score"],
            "privacy_score": privacy["privacy_score"],
            "benefits": benefits,
            "risks": risks,
            "recommendation": recommendation,
            "similarity_details": similarity,
            "privacy_details": privacy,
        }

    def _detect_feature_type(self, series: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(series):
            return "Boolean"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "Datetime"
        if pd.api.types.is_numeric_dtype(series):
            return "Numerical"
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            unique_ratio = series.nunique(dropna=True) / max(1, len(series))
            if unique_ratio > 0.9 and series.nunique(dropna=True) > 5:
                return "Identifier"
            if series.dropna().astype(str).str.len().mean() > 20:
                return "Text"
            return "Categorical"
        return "Categorical"

    def _find_target_candidates(self, df: pd.DataFrame, feature_types: Dict[str, str]) -> List[str]:
        candidates: List[str] = []
        for col in df.columns:
            col_lower = col.lower()
            if any(hint in col_lower for hint in self.TARGET_COLUMN_HINTS):
                candidates.append(col)
        if candidates:
            return candidates

        for col, ft in feature_types.items():
            if ft == "Categorical" and df[col].nunique(dropna=True) <= 20:
                candidates.append(col)
        return candidates

    def _detect_class_imbalance(
        self, df: pd.DataFrame, candidates: List[str]
    ) -> Dict[str, Any]:
        if not candidates:
            return {"is_imbalanced": False, "details": "No target candidates found."}

        candidate = candidates[0]
        counts = df[candidate].value_counts(dropna=False)
        if counts.empty:
            return {"is_imbalanced": False, "details": "Target candidate has no values."}

        ratio = counts.max() / max(1, counts.min()) if len(counts) > 1 else float("inf")
        is_imbalanced = ratio >= 2.0
        return {
            "is_imbalanced": is_imbalanced,
            "target_column": candidate,
            "ratio": float(round(ratio, 2)) if ratio != float("inf") else None,
            "counts": counts.to_dict(),
        }

    def _generate_statistical(self, df: pd.DataFrame, num_samples: int) -> pd.DataFrame:
        synthetic = {}
        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                synthetic[col] = [None] * num_samples
                continue
            dtype = self._detect_feature_type(series)
            if dtype == "Numerical":
                mean = float(series.mean())
                std = float(series.std(ddof=0))
                if std <= 0.0:
                    values = [mean] * num_samples
                else:
                    values = np.random.default_rng(42).normal(loc=mean, scale=std, size=num_samples)
                    values = np.clip(values, float(series.min()), float(series.max()))
                synthetic[col] = [float(x) for x in values]
            elif dtype == "Categorical":
                dist = self._value_distribution(series)
                categories = list(dist.keys())
                probabilities = list(dist.values())
                values = np.random.default_rng(42).choice(categories, size=num_samples, p=probabilities)
                synthetic[col] = [str(x) for x in values]
            elif dtype == "Boolean":
                true_ratio = float(series.astype(bool).mean())
                values = np.random.default_rng(42).choice([True, False], size=num_samples, p=[true_ratio, 1 - true_ratio])
                synthetic[col] = values.tolist()
            else:
                sampled = series.astype(str).sample(n=num_samples, replace=True, random_state=42).tolist()
                synthetic[col] = sampled
        return pd.DataFrame(synthetic)

    def _generate_balanced(self, df: pd.DataFrame, num_samples: int) -> pd.DataFrame:
        candidates = self._find_target_candidates(df, {col: self._detect_feature_type(df[col]) for col in df.columns})
        if not candidates:
            return df.sample(n=num_samples, replace=True, random_state=42).reset_index(drop=True)

        target_col = candidates[0]
        if target_col not in df.columns:
            return df.sample(n=num_samples, replace=True, random_state=42).reset_index(drop=True)

        groups = [group for _, group in df.groupby(target_col, dropna=False)]
        if not groups:
            return df.sample(n=num_samples, replace=True, random_state=42).reset_index(drop=True)

        weights = [1 / max(1, len(group)) for group in groups]
        group_sizes = [max(1, int(np.round(num_samples / len(groups)))) for _ in groups]
        samples: List[pd.DataFrame] = []
        for group, size in zip(groups, group_sizes):
            if group.empty:
                continue
            samples.append(group.sample(n=size, replace=True, random_state=42))
        synthetic = pd.concat(samples, ignore_index=True)
        if len(synthetic) < num_samples:
            additional = df.sample(n=num_samples - len(synthetic), replace=True, random_state=42)
            synthetic = pd.concat([synthetic, additional], ignore_index=True)
        return synthetic.sample(frac=1.0, random_state=42).reset_index(drop=True).iloc[:num_samples]

    def _estimate_quality(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame, method: str) -> float:
        base = 70.0
        if method == "bootstrap":
            base = 80.0
        elif method == "balanced":
            base = 75.0

        if set(original_df.columns) != set(synthetic_df.columns):
            base -= 20
        if len(synthetic_df) != len(original_df):
            base -= 5
        return max(0.0, min(100.0, base))

    def _range_overlap(self, orig: pd.Series, synth: pd.Series) -> float:
        orig_min, orig_max = float(orig.min()), float(orig.max())
        synth_min, synth_max = float(synth.min()), float(synth.max())
        if orig_max <= orig_min or synth_max <= synth_min:
            return 0.0
        overlap_low = max(orig_min, synth_min)
        overlap_high = min(orig_max, synth_max)
        if overlap_high <= overlap_low:
            return 0.0
        overlap = (overlap_high - overlap_low) / max(orig_max - orig_min, synth_max - synth_min)
        return float(round(overlap, 4))

    def _value_distribution(self, series: pd.Series) -> Dict[Any, float]:
        counts = series.astype(str).value_counts(dropna=False)
        total = float(counts.sum())
        return {value: float(count / total) for value, count in counts.items()}

    def _distribution_similarity(self, dist1: Dict[Any, float], dist2: Dict[Any, float]) -> float:
        keys = set(dist1) | set(dist2)
        score = 0.0
        for key in keys:
            score += abs(dist1.get(key, 0.0) - dist2.get(key, 0.0))
        return max(0.0, min(1.0, 1.0 - score / 2.0))

    def _count_near_duplicates(self, original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> int:
        numeric_cols = [col for col in original_df.columns if pd.api.types.is_numeric_dtype(original_df[col])]
        if not numeric_cols:
            return 0
        original_small = original_df[numeric_cols].fillna(0).astype(float)
        synthetic_small = synthetic_df[numeric_cols].fillna(0).astype(float)
        count = 0
        for _, row in synthetic_small.iterrows():
            distances = np.linalg.norm(original_small.values - row.values, axis=1)
            if np.any(distances <= 1.0):
                count += 1
        return count

    def _find_unique_identifier_columns(self, df: pd.DataFrame) -> List[str]:
        identifiers: List[str] = []
        for col in df.columns:
            series = df[col]
            unique_ratio = series.nunique(dropna=True) / max(1, len(series))
            if unique_ratio > 0.9 and series.nunique(dropna=True) > 10:
                identifiers.append(col)
        return identifiers

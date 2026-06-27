"""
Unified Trust Score Calculator

Provides single authoritative trust score calculation across all pipeline stages.
Ensures consistent numeric values (0-100) with no string fallbacks.
"""

import logging
from typing import Dict, Any, Optional

from utils.safe_checks import coerce_numeric_score

_logger = logging.getLogger(__name__)


def _normalize_score_component(value: Any, default: float = 50.0) -> float:
    """Normalize a score to 0-100 without ambiguous pandas truthiness."""
    numeric = coerce_numeric_score(value)
    if numeric is None:
        return default
    if 0 < numeric <= 1.0:
        numeric *= 100.0
    return min(100.0, max(0.0, float(numeric)))


class TrustScoreCalculator:
    """Calculates unified trust score from available components."""
    
    # Default weights if all components available
    DEFAULT_WEIGHTS = {
        "model_performance": 0.25,
        "explainability": 0.20,
        "fairness": 0.20,
        "privacy": 0.15,
        "deployment": 0.20,
    }
    
    @staticmethod
    def calculate_trust_score(
        model_reliability: Optional[float] = None,
        dataset_health: Optional[float] = None,
        fairness_score: Optional[float] = None,
        privacy_score: Optional[float] = None,
        deployment_readiness: Optional[Dict[str, Any]] = None,
        model_results: Optional[Dict[str, Any]] = None,
        explainability_available: bool = False,
        model_performance: Optional[float] = None,
    ) -> float:
        """
        Calculate unified trust score from available components.

        Args:
            model_reliability: Model reliability score 0-100 or 0-1.
            dataset_health: Dataset health score 0-100 or 0-1.
            fairness_score: Fairness assessment score 0-100.
            privacy_score: Privacy safety score 0-100.
            deployment_readiness: Deployment readiness dict with risk_level.
            model_results: Model results dict for fallback score extraction.
            explainability_available: Whether SHAP/feature importance is available.
            model_performance: Alias for model_reliability for backward compatibility.

        Returns:
            float: Trust score 0-100 (never None, "", "-", or "Not Evaluated").
        """

        if model_reliability is None and model_performance is not None:
            model_reliability = model_performance

        scores: Dict[str, float] = {}
        # Force robust fallback weights for partially missing components
        weights = {
            "model_reliability": 0.40,
            "dataset_health": 0.20,
            "deployment": 0.20,
            "fairness": 0.10,
            "privacy": 0.10,
        }

        # Model reliability fallback
        if model_reliability is not None:
            scores["model_reliability"] = _normalize_score_component(model_reliability)
        elif isinstance(model_results, dict) and model_results.get("best_score") is not None:
            scores["model_reliability"] = _normalize_score_component(model_results.get("best_score"))
        else:
            scores["model_reliability"] = 50.0

        # Dataset health fallback
        scores["dataset_health"] = _normalize_score_component(dataset_health)

        # Fairness fallback
        scores["fairness"] = _normalize_score_component(fairness_score)

        # Privacy fallback
        scores["privacy"] = _normalize_score_component(privacy_score)

        # Deployment readiness fallback
        scores["deployment"] = TrustScoreCalculator._calculate_deployment_score(deployment_readiness)

        # If explainability is available and trust was otherwise neutral, boost model reliability slightly
        if explainability_available and scores["model_reliability"] == 50.0:
            scores["model_reliability"] = min(100.0, scores["model_reliability"] + 5.0)

        weighted_sum = sum(scores[key] * weights[key] for key in weights)
        trust_score = weighted_sum / sum(weights.values())
        trust_score = float(trust_score)
        trust_score = min(100.0, max(0.0, trust_score))

        try:
            _logger.info(f"Final Trust Score: {trust_score}")
            _logger.info(f"Trust Components: {scores}")
        except Exception:
            pass

        return round(trust_score, 1)
    
    @staticmethod
    def _calculate_deployment_score(deployment_readiness: Optional[Dict[str, Any]]) -> float:
        """
        Convert deployment readiness to trust score component.
        
        Args:
            deployment_readiness: Deployment dict with risk_level and other info
            
        Returns:
            float: Deployment score 0-100
        """
        if deployment_readiness is None or not isinstance(deployment_readiness, dict):
            return 50.0  # Neutral if no info

        risk_level = deployment_readiness.get("risk_level", "Unknown")
        if isinstance(risk_level, str):
            risk_level = risk_level.lower()

        # Map risk level to score
        risk_mapping = {
            "low": 85.0,
            "medium": 60.0,
            "high": 30.0,
            "unknown": 50.0,
        }
        
        deployment_score = risk_mapping.get(risk_level, 50.0)
        
        # Bonus for production readiness
        if deployment_readiness.get("is_production_ready") is True:
            deployment_score = min(100.0, deployment_score + 10)
        
        return deployment_score
    
    @staticmethod
    def ensure_numeric_trust_score(trust_score: Any) -> float:
        """
        Convert any trust score value to numeric.
        
        Handles:
        - String "Not Evaluated" → 0
        - None → 0
        - "-" → 0
        - Numeric strings → converted
        - Float/int → returned as-is
        
        Args:
            trust_score: Any value that might be a trust score
            
        Returns:
            float: Numeric trust score 0-100
        """
        if trust_score is None:
            return 0.0
        
        if isinstance(trust_score, str):
            lower_score = trust_score.lower().strip()
            if lower_score in ["not evaluated", "not applicable", "n/a", "-", "", "none"]:
                return 0.0
            try:
                return float(trust_score)
            except (ValueError, TypeError):
                return 0.0
        
        if isinstance(trust_score, (int, float)):
            value = float(trust_score)
            # Normalize if decimal (0.95 → 95)
            if 0 < value <= 1.0:
                value *= 100
            return min(100.0, max(0.0, value))
        
        return 0.0

    @staticmethod
    def deployment_status_from_trust_score(trust_score: float) -> str:
        """Map trust score to production deployment status."""
        if trust_score > 80:
            return "Production Ready"
        if trust_score >= 60:
            return "Needs Monitoring"
        return "Not Ready"

    @staticmethod
    def recommendation_from_trust_score(trust_score: float) -> str:
        """Generate a human-friendly recommendation from trust score."""
        if trust_score > 80:
            return "Deploy the selected model to production."
        if trust_score >= 60:
            return "Monitor model performance and continue refinement before full production deployment."
        return "Further development required before production deployment."


def create_executive_metrics_object(
    best_model: Optional[str] = None,
    model_version: Optional[str] = None,
    accuracy: Optional[float] = None,
    trust_score: Optional[float] = None,
    risk_level: Optional[str] = None,
    deployment_status: Optional[str] = None,
    deployment_readiness: Optional[Dict[str, Any]] = None,
    health_score: Optional[float] = None,
    confidence_score: Optional[float] = None,
    final_decision: Optional[Dict[str, Any]] = None,
    runtime_seconds: Optional[float] = None,
    model_results: Optional[Dict[str, Any]] = None,
    deployment_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create authoritative executive metrics object for display consistency.
    
    This object serves as SINGLE SOURCE OF TRUTH for all UI pages:
    - Home Dashboard
    - Reports Page
    - Chief Decision Panel
    - Executive Summary
    - PDF Export
    - Excel Export
    
    Args:
        best_model: Best model algorithm name (e.g., "Random Forest")
        model_version: Model version ID (e.g., "v1")
        accuracy: Model accuracy/F1/AUC score
        trust_score: Trust score 0-100
        risk_level: Deployment risk level (Low/Medium/High)
        deployment_status: Deployment status (ready/monitor/not_ready)
        deployment_readiness: Full deployment readiness dict
        health_score: Dataset health score 0-100
        confidence_score: Overall confidence score 0-100
        final_decision: Final deployment decision dict
        runtime_seconds: Pipeline execution time in seconds
        model_results: Full model results dict (for extraction)
        deployment_info: Full deployment info dict (for extraction)
        
    Returns:
        dict: Executive metrics object with guaranteed numeric values
    """
    
    # Extract best_model from model_results if not provided
    if not best_model and isinstance(model_results, dict):
        best_model = model_results.get("best_model")
    best_model = best_model or "—"

    # Extract accuracy if not provided
    if accuracy is None and isinstance(model_results, dict):
        accuracy = model_results.get("best_score")
        best_name = model_results.get("best_model")
        metrics = model_results.get("metrics") or {}
        if (accuracy is None or (isinstance(accuracy, (int, float)) and float(accuracy) <= 0)) and best_name and isinstance(metrics, dict):
            accuracy = metrics.get(best_name)
    if accuracy is not None:
        try:
            accuracy = float(accuracy)
            if accuracy <= 0:
                accuracy = None
            elif 0 < accuracy <= 1.0:
                accuracy = round(accuracy * 100, 2)
            else:
                accuracy = round(accuracy, 2)
        except (TypeError, ValueError):
            accuracy = None
    
    # Ensure trust_score is numeric
    trust_score = TrustScoreCalculator.ensure_numeric_trust_score(trust_score)
    
    # Extract risk_level from deployment_readiness if not provided
    if not risk_level and isinstance(deployment_readiness, dict):
        risk_level = deployment_readiness.get("risk_level", "Unknown")
    risk_level = risk_level or "Unknown"

    # Extract deployment_status if not provided
    if not deployment_status and isinstance(deployment_info, dict):
        deployment_status = deployment_info.get("deployment_status")
    deployment_status = deployment_status or "—"
    
    # Ensure health_score is numeric
    health_numeric = coerce_numeric_score(health_score)
    if health_numeric is not None and 0 < health_numeric <= 1.0:
        health_score = round(health_numeric * 100, 2)
    elif health_numeric is not None:
        health_score = round(health_numeric, 2)
    else:
        health_score = 0.0
    
    # Ensure confidence_score is numeric
    confidence_numeric = coerce_numeric_score(confidence_score)
    if confidence_numeric is not None and 0 < confidence_numeric <= 1.0:
        confidence_score = round(confidence_numeric * 100, 2)
    elif confidence_numeric is not None:
        confidence_score = round(confidence_numeric, 2)
    else:
        confidence_score = 0.0
    
    # Ensure runtime_seconds is numeric
    runtime_seconds = runtime_seconds or 0.0
    
    # Build final_decision if not provided
    if not final_decision:
        final_decision = {
            "risk_level": risk_level,
            "deployment_status": deployment_status,
            "trust_score": trust_score,
            "confidence_score": confidence_score,
            "recommendation": f"Risk: {risk_level} | Trust: {trust_score}",
        }
    
    # Create authoritative metrics object
    executive_metrics = {
        "best_model": best_model,
        "model_version": model_version or "v1",
        "accuracy": round(float(accuracy), 2) if accuracy is not None else None,
        "trust_score": round(float(trust_score), 1),
        "risk_level": risk_level,
        "deployment_status": deployment_status,
        "deployment_readiness": deployment_readiness or {},
        # Authoritative deployment decision object
        "deployment_decision": {
            "deployment_status": deployment_status,
            "readiness_score": TrustScoreCalculator._calculate_deployment_score(deployment_readiness),
            "risk_level": risk_level,
            "final_decision": final_decision,
            "recommendation": final_decision.get("recommendation"),
        },
        "health_score": round(float(health_score), 1) if health_score else 0.0,
        "confidence_score": round(float(confidence_score), 1) if confidence_score else 0.0,
        "final_decision": final_decision,
        "runtime_seconds": round(float(runtime_seconds), 2) if runtime_seconds else 0.0,
    }
    
    return executive_metrics

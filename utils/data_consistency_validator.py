"""
Data Consistency Validator

Validates that all executive metrics are consistent across:
- Home Dashboard
- Reports Page
- Chief Decision Panel
- PDF Export
- Excel Export

Ensures single source of truth is maintained.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

_logger = logging.getLogger(__name__)


class DataConsistencyValidator:
    """Validates executive metrics consistency across all pages and exports."""
    
    CRITICAL_METRICS = [
        "best_model",
        "trust_score",
        "health_score",
        "deployment_status",
        "final_decision",
    ]
    
    @staticmethod
    def validate_executive_metrics(executive_metrics: Optional[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate that executive_metrics object has all required fields.
        
        Args:
            executive_metrics: Executive metrics object from pipeline
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not executive_metrics or not isinstance(executive_metrics, dict):
            return False, ["Executive metrics object is missing or invalid"]
        
        required_keys = [
            "best_model",
            "model_version",
            "accuracy",
            "trust_score",
            "risk_level",
            "deployment_status",
            "health_score",
            "confidence_score",
            "final_decision",
            "runtime_seconds",
        ]
        
        for key in required_keys:
            if key not in executive_metrics:
                errors.append(f"Missing required key: {key}")
        
        # Validate data types
        if executive_metrics.get("best_model") and not isinstance(executive_metrics.get("best_model"), str):
            errors.append("best_model must be a string")
        
        if executive_metrics.get("trust_score") is not None:
            if not isinstance(executive_metrics.get("trust_score"), (int, float)):
                errors.append("trust_score must be numeric")
            else:
                trust = float(executive_metrics.get("trust_score"))
                if not (0 <= trust <= 100):
                    errors.append(f"trust_score out of range: {trust} (must be 0-100)")
        
        if executive_metrics.get("health_score") is not None:
            if not isinstance(executive_metrics.get("health_score"), (int, float)):
                errors.append("health_score must be numeric")
            else:
                health = float(executive_metrics.get("health_score"))
                if not (0 <= health <= 100):
                    errors.append(f"health_score out of range: {health} (must be 0-100)")
        
        if executive_metrics.get("confidence_score") is not None:
            if not isinstance(executive_metrics.get("confidence_score"), (int, float)):
                errors.append("confidence_score must be numeric")
        
        # Validate deployment_status
        valid_status = ["Production Ready", "Needs Monitoring", "Not Ready"]
        if executive_metrics.get("deployment_status") not in valid_status:
            errors.append(f"deployment_status must be one of: {valid_status}, got {executive_metrics.get('deployment_status')}")
        
        # Validate risk_level
        valid_risks = ["Low", "Medium", "High", "Unknown"]
        if executive_metrics.get("risk_level") not in valid_risks:
            errors.append(f"risk_level must be one of: {valid_risks}, got {executive_metrics.get('risk_level')}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_no_string_nulls(executive_metrics: Optional[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate that no critical fields contain string nulls like "Not Evaluated", "-", "None".
        
        Args:
            executive_metrics: Executive metrics object
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not executive_metrics or not isinstance(executive_metrics, dict):
            return True, []  # No metrics to validate
        
        forbidden_values = [
            "Not Evaluated",
            "not evaluated",
            "-",
            "None",
            "none",
            "N/A",
            "n/a",
            "Unknown",
            "unknown",
            "",
        ]
        
        critical_numeric_fields = [
            "trust_score",
            "health_score",
            "accuracy",
            "confidence_score",
        ]
        
        for field in critical_numeric_fields:
            value = executive_metrics.get(field)
            if value is not None and isinstance(value, str):
                if value in forbidden_values:
                    errors.append(
                        f"{field} contains forbidden string value: '{value}' "
                        f"(must be numeric 0-100)"
                    )
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_consistency_across_pages(
        dashboard_metrics: Dict[str, Any],
        reports_metrics: Dict[str, Any],
        chief_metrics: Dict[str, Any],
        pdf_metrics: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Validate that all pages display identical values for critical metrics.
        
        Args:
            dashboard_metrics: Metrics from Home Dashboard
            reports_metrics: Metrics from Reports Page
            chief_metrics: Metrics from Chief Decision Panel
            pdf_metrics: Metrics from PDF Export
            
        Returns:
            Tuple of (is_consistent, list_of_mismatches)
        """
        mismatches = []
        
        # Fields to compare
        critical_fields = [
            ("best_model", "Model name"),
            ("trust_score", "Trust score"),
            ("health_score", "Health score"),
            ("deployment_status", "Deployment status"),
            ("risk_level", "Risk level"),
        ]
        
        pages = {
            "Dashboard": dashboard_metrics,
            "Reports": reports_metrics,
            "Chief Decision": chief_metrics,
            "PDF": pdf_metrics,
        }
        
        for field, label in critical_fields:
            values = {}
            for page, metrics in pages.items():
                if isinstance(metrics, dict) and field in metrics:
                    values[page] = metrics[field]
            
            if not values:
                continue  # Field not present on any page
            
            # Check if all values match
            if len(set(str(v) for v in values.values())) > 1:
                value_str = ", ".join(f"{page}={val}" for page, val in values.items())
                mismatches.append(
                    f"{label} mismatch: {value_str} "
                    f"(all pages must display identical value)"
                )
        
        return len(mismatches) == 0, mismatches
    
    @staticmethod
    def validate_numeric_conversions(executive_metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that all numeric fields are properly converted (never decimals when 0-100 range expected).
        
        Args:
            executive_metrics: Executive metrics object
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        numeric_fields = ["trust_score", "health_score", "accuracy", "confidence_score"]
        
        for field in numeric_fields:
            value = executive_metrics.get(field)
            if value is None:
                continue
            
            if not isinstance(value, (int, float)):
                errors.append(f"{field} is not numeric: {type(value).__name__}")
                continue
            
            # Check if value should be normalized (0-1 range)
            if 0 < value <= 1.0 and field in ["accuracy", "trust_score", "health_score"]:
                errors.append(
                    f"{field} appears to be decimal ({value}), "
                    f"should be 0-100 range"
                )
        
        return len(errors) == 0, errors
    
    @staticmethod
    def log_validation_report(
        executive_metrics: Dict[str, Any],
        is_valid: bool,
        errors: List[str],
    ) -> None:
        """Log a comprehensive validation report."""
        if is_valid:
            _logger.info(
                f"[DataConsistency] ✓ All executive metrics validated successfully"
            )
            if executive_metrics:
                _logger.info(
                    f"[DataConsistency] Metrics Summary: "
                    f"best_model={executive_metrics.get('best_model')}, "
                    f"trust_score={executive_metrics.get('trust_score')}, "
                    f"health_score={executive_metrics.get('health_score')}, "
                    f"deployment_status={executive_metrics.get('deployment_status')}, "
                    f"risk_level={executive_metrics.get('risk_level')}"
                )
        else:
            _logger.error(
                f"[DataConsistency] ✗ Validation failed with {len(errors)} errors:"
            )
            for error in errors:
                _logger.error(f"  - {error}")
    
    @staticmethod
    def validate_all(executive_metrics: Dict[str, Any]) -> bool:
        """
        Run all validation checks on executive_metrics.
        
        Args:
            executive_metrics: Executive metrics object
            
        Returns:
            True if all validations pass, False otherwise
        """
        all_errors = []
        
        # Check required fields
        valid_fields, field_errors = DataConsistencyValidator.validate_executive_metrics(
            executive_metrics
        )
        all_errors.extend(field_errors)
        
        # Check no string nulls
        valid_nulls, null_errors = DataConsistencyValidator.validate_no_string_nulls(
            executive_metrics
        )
        all_errors.extend(null_errors)
        
        # Check numeric conversions
        valid_conversions, conversion_errors = DataConsistencyValidator.validate_numeric_conversions(
            executive_metrics
        )
        all_errors.extend(conversion_errors)
        
        is_valid = len(all_errors) == 0
        
        DataConsistencyValidator.log_validation_report(
            executive_metrics,
            is_valid,
            all_errors,
        )
        
        return is_valid

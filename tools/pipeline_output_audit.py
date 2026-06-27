#!/usr/bin/env python
"""
Complete end-to-end pipeline audit:
Traces all pipeline output keys and verifies UI mappings.
"""

import json
import sys
import pandas as pd
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.master_autonomous_pipeline import MasterAutonomousPipeline
from utils.pipeline_bridge import normalize_pipeline_output, apply_autonomous_result_to_session

def audit_pipeline_output():
    """
    Audit complete pipeline output structure.
    """
    print("\n" + "="*80)
    print("PIPELINE OUTPUT AUDIT - WINE QUALITY DATASET")
    print("="*80 + "\n")
    
    # Load Wine Quality dataset
    df = pd.read_csv("data/Wine-Quality.csv")
    print(f"✓ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Initialize pipeline
    pipeline = MasterAutonomousPipeline()
    print("✓ Initialized MasterAutonomousPipeline")
    
    # Run pipeline with smart mode and progress tracking
    print("\n→ Running pipeline (this will take a few minutes)...\n")
    
    output = pipeline.run_pipeline(
        dataset=df,
        dataset_name="Wine-Quality",
        project_goal="Predict wine quality scores",
        constraints={},
        smart_mode=True,
        max_seconds=300,
        progress_callback=lambda evt: print(
            f"  [{evt.get('percent')}%] {evt.get('stage')}: {evt.get('status')}"
        ),
        use_cache=False,
    )
    
    print("\n✓ Pipeline completed\n")
    
    # AUDIT TABLE 1: PIPELINE OUTPUT KEYS
    print("="*80)
    print("AUDIT TABLE 1: PIPELINE OUTPUT STRUCTURE")
    print("="*80 + "\n")
    
    critical_keys = [
        "dataset_report",
        "cleaning_results",
        "eda_results",
        "feature_engineering_results",
        "model_results",
        "ai_trust_results",
        "deployment_readiness",
        "final_score",
        "final_report",
        "recommendation",
    ]
    
    print(f"{'Key':<40} | {'Present':<10} | {'Type':<20} | {'Has Data':<10}")
    print("-" * 90)
    
    for key in critical_keys:
        present = key in output
        value = output.get(key)
        val_type = type(value).__name__
        has_data = "✓ Yes" if value else "✗ No"
        
        status = "✓ PASS" if present else "✗ FAIL"
        print(f"{key:<40} | {status:<10} | {val_type:<20} | {has_data:<10}")
    
    # AUDIT TABLE 2: CRITICAL OUTPUT VALUES
    print("\n" + "="*80)
    print("AUDIT TABLE 2: CRITICAL OUTPUT VALUES")
    print("="*80 + "\n")
    
    model_results = output.get("model_results", {})
    trust_results = output.get("ai_trust_results", {})
    deploy_results = output.get("deployment_readiness", {})
    final_report = output.get("final_report", {})
    
    print(f"{'Field':<40} | {'Value':<40}")
    print("-" * 85)
    
    critical_values = [
        ("best_model", model_results.get("best_model", "MISSING")),
        ("model_score", model_results.get("model_readiness_score", "MISSING")),
        ("best_metrics", str(model_results.get("metrics", {}))[:38]),
        ("trust_score", trust_results.get("trust_score", "MISSING")),
        ("fairness_score", trust_results.get("fairness_score", "MISSING")),
        ("deployment_risk_level", deploy_results.get("risk_level", "MISSING")),
        ("final_score", output.get("final_score", "MISSING")),
        ("recommendation", output.get("recommendation", "MISSING")),
        ("final_report.path", final_report.get("path", "MISSING") if isinstance(final_report, dict) else "NOT_DICT"),
        ("report_has_payload", "✓ YES" if isinstance(final_report, dict) and final_report.get("payload") else "✗ NO"),
    ]
    
    for field, value in critical_values:
        value_str = str(value)[:38]
        status = "✓" if value != "MISSING" and value != "NOT_DICT" else "✗"
        print(f"{status} {field:<38} | {value_str:<40}")
    
    # AUDIT TABLE 3: PIPELINE OUTPUT → UI MAPPING
    print("\n" + "="*80)
    print("AUDIT TABLE 3: PIPELINE OUTPUT → UI MAPPING")
    print("="*80 + "\n")
    
    ui_mappings = [
        ("best_model", "model_results.best_model", model_results.get("best_model")),
        ("best_model_alt", "model_results['metrics'] keys", list(model_results.get("metrics", {}).keys())[:3]),
        ("trust_score", "ai_trust_results.trust_score", trust_results.get("trust_score")),
        ("deployment_status", "deployment_readiness.risk_level", deploy_results.get("risk_level")),
        ("final_score", "output.final_score", output.get("final_score")),
        ("final_score_alt", "output.final_scores.overall_score", output.get("final_scores", {}).get("overall_score")),
        ("report_path", "final_report.path", final_report.get("path") if isinstance(final_report, dict) else None),
    ]
    
    print(f"{'UI Card':<25} | {'Source Path':<40} | {'Match':<20}")
    print("-" * 90)
    
    for ui_name, source_path, value in ui_mappings:
        match_status = "✓ PASS" if value else "✗ FAIL"
        print(f"{ui_name:<25} | {source_path:<40} | {match_status:<20}")
    
    # AUDIT TABLE 4: NORMALIZED OUTPUT ALIASES
    print("\n" + "="*80)
    print("AUDIT TABLE 4: NORMALIZE_PIPELINE_OUTPUT ALIASES")
    print("="*80 + "\n")
    
    normalized = normalize_pipeline_output(output)
    
    aliases = [
        ("dataset_report", "dataset_analysis", "dataset_profile"),
        ("model_results", "automl_results"),
        ("explainability_results", "xai_results", "shap_results"),
        ("ai_trust_results", "ethics_report", "trust_results"),
        ("deployment_readiness", "deployment_results"),
        ("final_report", "executive_report"),
    ]
    
    print(f"{'Primary Key':<30} | {'Aliases':<50}")
    print("-" * 85)
    
    for alias_group in aliases:
        primary = alias_group[0]
        alts = ", ".join(alias_group[1:])
        primary_val = normalized.get(primary)
        has_value = "✓" if primary_val else "✗"
        print(f"{has_value} {primary:<28} | {alts:<50}")
    
    # AUDIT TABLE 5: SESSION STATE PERSISTENCE
    print("\n" + "="*80)
    print("AUDIT TABLE 5: SESSION STATE PERSISTENCE TARGETS")
    print("="*80 + "\n")
    
    # Simulate what apply_autonomous_result_to_session would persist
    persistence_targets = [
        ("SessionKeys.AUTONOMOUS_RESULT", output, "Full pipeline output dict"),
        ("SessionKeys.BEST_MODEL_NAME", model_results.get("best_model"), "From model_results"),
        ("SessionKeys.REPORT_PATH", final_report.get("path") if isinstance(final_report, dict) else None, "From final_report.path"),
        ("SessionKeys.REPORT_PAYLOAD", final_report.get("payload") if isinstance(final_report, dict) else None, "From final_report.payload"),
        ("SessionKeys.REPORT_GENERATED", True if isinstance(final_report, dict) and final_report.get("path") else False, "Report path exists"),
        ("SessionKeys.CONFIDENCE_SCORE", trust_results.get("trust_score"), "From ai_trust_results.trust_score"),
    ]
    
    print(f"{'Session Key':<35} | {'Value Type':<20} | {'Source':<30}")
    print("-" * 90)
    
    for key, value, source in persistence_targets:
        val_type = type(value).__name__ if value else "None"
        has_value = "✓" if value else "✗"
        print(f"{has_value} {key:<33} | {val_type:<20} | {source:<30}")
    
    # Final verdict
    print("\n" + "="*80)
    print("FINAL AUDIT VERDICT")
    print("="*80 + "\n")
    
    checks = {
        "Pipeline produces best_model": bool(model_results.get("best_model")),
        "Pipeline produces trust_score": trust_results.get("trust_score") is not None,
        "Pipeline produces final_score": output.get("final_score") is not None,
        "Pipeline produces deployment_readiness": bool(deploy_results),
        "Pipeline produces final_report": isinstance(final_report, dict) and "path" in final_report,
        "final_report has path value": isinstance(final_report, dict) and final_report.get("path") is not None,
        "final_report has payload": isinstance(final_report, dict) and isinstance(final_report.get("payload"), dict),
    }
    
    for check, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check}")
    
    all_pass = all(checks.values())
    print(f"\n{'='*80}")
    if all_pass:
        print("✓ ALL CHECKS PASSED - Pipeline output is complete and correct")
    else:
        print("✗ SOME CHECKS FAILED - See details above")
    print(f"{'='*80}\n")
    
    # Write audit output to file
    audit_file = Path("storage/logs/pipeline_audit.json")
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    
    audit_data = {
        "dataset": "Wine-Quality",
        "pipeline_complete": all_pass,
        "critical_keys": {k: (k in output) for k in critical_keys},
        "critical_values": {
            "best_model": model_results.get("best_model"),
            "trust_score": trust_results.get("trust_score"),
            "final_score": output.get("final_score"),
            "deployment_risk": deploy_results.get("risk_level"),
        },
        "checks": checks,
    }
    
    with open(audit_file, "w") as f:
        json.dump(audit_data, f, indent=2, default=str)
    
    print(f"✓ Audit written to {audit_file}\n")
    
    return all_pass


if __name__ == "__main__":
    try:
        success = audit_pipeline_output()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Audit failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

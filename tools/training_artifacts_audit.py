#!/usr/bin/env python
"""
Detailed training_artifacts audit - checks if artifacts are correctly passed to session state
"""

import json
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.master_autonomous_pipeline import MasterAutonomousPipeline
from utils.pipeline_bridge import normalize_pipeline_output

def audit_training_artifacts():
    """
    Audit training_artifacts in pipeline output.
    """
    print("\n" + "="*80)
    print("TRAINING ARTIFACTS AUDIT")
    print("="*80 + "\n")
    
    # Load dataset
    df = pd.read_csv("data/Wine-Quality.csv")
    print(f"✓ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # Run pipeline
    pipeline = MasterAutonomousPipeline()
    print("→ Running pipeline...\n")
    
    output = pipeline.run_pipeline(
        dataset=df,
        dataset_name="Wine-Quality",
        project_goal="Predict wine quality scores",
        constraints={},
        smart_mode=True,
        max_seconds=300,
        progress_callback=lambda evt: None,
        use_cache=False,
    )
    
    print("✓ Pipeline completed\n")
    
    # AUDIT: Check training_artifacts structure
    print("="*80)
    print("TRAINING ARTIFACTS STRUCTURE")
    print("="*80 + "\n")
    
    artifacts = output.get("training_artifacts", {})
    print(f"training_artifacts present: {'✓ YES' if artifacts else '✗ NO'}")
    print(f"training_artifacts type: {type(artifacts).__name__}\n")
    
    artifact_keys = [
        "best_model",
        "best_name",
        "results",
        "problem_type",
        "target_column",
        "X_data",
        "y_data",
        "target_encoder",
        "extras",
        "cleaned_dataframe",
    ]
    
    print(f"{'Key':<30} | {'Present':<10} | {'Type':<20} | {'Value':<30}")
    print("-" * 95)
    
    for key in artifact_keys:
        value = artifacts.get(key)
        present = "✓ YES" if value is not None else "✗ NO"
        val_type = type(value).__name__
        val_str = str(value)[:28] if value is not None else "None"
        print(f"{key:<30} | {present:<10} | {val_type:<20} | {val_str:<30}")
    
    # AUDIT: Check apply_autonomous_result_to_session logic
    print("\n" + "="*80)
    print("STORE_MODEL_RESULTS ELIGIBILITY CHECK")
    print("="*80 + "\n")
    
    best_model = artifacts.get("best_model")
    best_name = artifacts.get("best_name")
    results = artifacts.get("results") or {}
    X = artifacts.get("X_data")
    y = artifacts.get("y_data")
    
    print(f"best_model is not None: {best_model is not None} (value: {best_model})")
    print(f"best_name is not None and truthy: {bool(best_name)} (value: {best_name})")
    print(f"results is not None and truthy: {bool(results)} (value: {len(results)} items)")
    print(f"X is not None: {X is not None}")
    print(f"y is not None: {y is not None}")
    
    store_eligible = best_model is not None and best_name and results
    print(f"\nstore_model_results ELIGIBLE: {store_eligible}")
    
    if not store_eligible:
        print("\n✗ WARNING: store_model_results will NOT be called!")
        print("This means BEST_MODEL_NAME and BEST_SCORE will not be set in session state")
    else:
        print("\n✓ store_model_results will be called correctly")
    
    # AUDIT: Check if best_name is in results
    if best_name and isinstance(results, dict):
        in_results = best_name in results
        print(f"\nbest_name ('{best_name}') in results: {in_results}")
        if in_results:
            print(f"  → BEST_SCORE would be: {results[best_name]}")
        else:
            print(f"  → Available keys in results: {list(results.keys())[:5]}...")
    
    # AUDIT: Alternative pathway check
    print("\n" + "="*80)
    print("ALTERNATIVE PATHWAY (model_results)")
    print("="*80 + "\n")
    
    model_results = output.get("model_results", {})
    alt_best_model = model_results.get("best_model")
    alt_results = model_results.get("metrics", {})
    
    print(f"model_results.best_model: {alt_best_model}")
    print(f"model_results.metrics: {len(alt_results)} items")
    
    if alt_best_model and isinstance(alt_results, dict):
        if alt_best_model in alt_results:
            print(f"  → model_results score for best model: {alt_results[alt_best_model]}")
    
    # AUDIT: normalize_pipeline_output
    print("\n" + "="*80)
    print("AFTER normalize_pipeline_output()")
    print("="*80 + "\n")
    
    normalized = normalize_pipeline_output(output)
    norm_artifacts = normalized.get("training_artifacts", {})
    norm_best_name = norm_artifacts.get("best_name")
    norm_results = norm_artifacts.get("results", {})
    
    print(f"Normalized training_artifacts.best_name: {norm_best_name}")
    print(f"Normalized training_artifacts.results: {len(norm_results)} items")
    
    # FINAL VERDICT
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80 + "\n")
    
    checks = {
        "training_artifacts present": bool(artifacts),
        "artifacts has best_model": best_model is not None,
        "artifacts has best_name": bool(best_name),
        "artifacts has results": bool(results),
        "store_model_results will execute": store_eligible,
        "best_name in results": best_name in results if best_name and results else False,
    }
    
    for check, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check}")
    
    all_pass = all(checks.values())
    print(f"\n{'='*80}")
    if all_pass:
        print("✓ ALL CHECKS PASSED - artifacts will be correctly persisted")
    else:
        failed = [c for c, r in checks.items() if not r]
        print(f"✗ CHECKS FAILED - {len(failed)} issues found:")
        for issue in failed:
            print(f"  - {issue}")
    print(f"{'='*80}\n")
    
    return all_pass


if __name__ == "__main__":
    try:
        success = audit_training_artifacts()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Audit failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

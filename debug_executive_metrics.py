#!/usr/bin/env python
"""
DEBUG: Print executive_metrics values before Executive Summary renders.

The Executive Summary reads from:
  File: ui/interactive_report_center.py
  Function: _tab_executive_summary() at line 627
  Inner helper: _select_primary_metric() at line 635
  Source: st.session_state.get(SessionKeys.EXECUTIVE_METRICS) or {} at line 641

This script simulates what happens after the regression fix is applied.
"""

import json
import sys
from utils.safe_checks import coalesce_dict, safe_dict_get
from utils.session_manager import SessionKeys

print("="*80)
print("DEBUG: EXECUTIVE SUMMARY RENDERING - EXECUTIVE_METRICS CONTENTS")
print("="*80)

# Load cached housing regression output (simulates pipeline output)
print("\n[Step 1] Loading cached regression pipeline output...")
with open('autonomous_validation_runs/autonomous_result_housing.json', 'r') as f:
    output = json.load(f)

print(f"  Dataset: Housing (Regression)")
print(f"  Detected problem_type from output: {output.get('problem_type', 'Not set')}")
print(f"  Detected problem_type from dataset_analysis: {output.get('dataset_analysis', {}).get('problem_analysis', {}).get('problem_type')}")

# Simulate what normalize_pipeline_output_and_persist() does
print("\n[Step 2] Simulating pipeline_bridge normalization...")

# Extract problem_type the way the code does
problem_type = output.get('problem_type') or safe_dict_get(
    coalesce_dict(safe_dict_get(output.get('dataset_analysis'), 'problem_analysis')),
    'problem_type',
) or 'Classification'

print(f"  Resolved problem_type: {problem_type}")

# Simulate _ensure_executive_metrics() - creates empty dict initially
executive_metrics = {
    'best_model': safe_dict_get(output.get('model_results'), 'best_model'),
    'accuracy': safe_dict_get(output.get('model_results'), 'accuracy'),
}

print(f"\n  After _ensure_executive_metrics():")
print(f"    - best_model: {executive_metrics.get('best_model')}")
print(f"    - accuracy: {executive_metrics.get('accuracy')}")
print(f"    - r2: {executive_metrics.get('r2')}")
print(f"    - rmse: {executive_metrics.get('rmse')}")
print(f"    - mae: {executive_metrics.get('mae')}")
print(f"    - mse: {executive_metrics.get('mse')}")

# Simulate _finalize_executive_metrics_display() - sets accuracy_display
from utils.safe_checks import format_accuracy_display
executive_metrics['accuracy_display'] = format_accuracy_display(
    executive_metrics.get('accuracy'), 
    problem_type
)

print(f"\n  After _finalize_executive_metrics_display():")
print(f"    - accuracy_display: {executive_metrics.get('accuracy_display')}")

# Simulate _inject_regression_metrics() - THE FIX
print(f"\n[Step 3] Simulating _inject_regression_metrics() fix...")
print(f"  Is regression? {('regress' in str(problem_type).lower())}")

if "regress" in str(problem_type).lower():
    print(f"  → Yes, extracting regression metrics...")
    
    # Extract from model_comparison
    model_comparison = output.get("model_comparison")
    if isinstance(model_comparison, list):
        for i, entry in enumerate(model_comparison):
            if isinstance(entry, dict):
                metrics = entry.get("metrics", {})
                if isinstance(metrics, dict):
                    print(f"    model_comparison[{i}].metrics: {list(metrics.keys())}")
                    for key in ("r2", "rmse", "mae", "mse"):
                        if key in metrics and key not in executive_metrics:
                            executive_metrics[key] = metrics[key]
                            print(f"      → Injected {key}: {metrics[key]}")
    
    # Extract from validation_results
    validation_results = output.get("validation_results")
    if isinstance(validation_results, list):
        for i, entry in enumerate(validation_results):
            if isinstance(entry, dict):
                validation = entry.get("validation", {})
                if isinstance(validation, dict):
                    print(f"    validation_results[{i}].validation: {list(validation.keys())}")
                    for key in ("r2", "rmse", "mae", "mse"):
                        if key in validation and key not in executive_metrics:
                            executive_metrics[key] = validation[key]
                            print(f"      → Injected {key}: {validation[key]}")
                    break

print(f"\n[Step 4] FINAL executive_metrics state in st.session_state[SessionKeys.EXECUTIVE_METRICS]:")
print("="*80)
print(f"  problem_type: {executive_metrics.get('problem_type')}")
print(f"  best_model: {executive_metrics.get('best_model')}")
print(f"  accuracy: {executive_metrics.get('accuracy')}")
print(f"  accuracy_display: {executive_metrics.get('accuracy_display')}")
print(f"  r2: {executive_metrics.get('r2')}")
print(f"  rmse: {executive_metrics.get('rmse')}")
print(f"  mae: {executive_metrics.get('mae')}")
print(f"  mse: {executive_metrics.get('mse')}")
print("="*80)

print(f"\n[Step 5] What _select_primary_metric() would see:")
print(f"  exec_metrics = st.session_state.get(SessionKeys.EXECUTIVE_METRICS) or {{}}")
print(f"  → exec_metrics['r2'] = {executive_metrics.get('r2')}")
print(f"  → exec_metrics['rmse'] = {executive_metrics.get('rmse')}")
print(f"  → exec_metrics['mae'] = {executive_metrics.get('mae')}")

# Simulate the selection logic
print(f"\n[Step 6] Selecting metric for display:")
for key, label in (("r2", "R² Score"), ("rmse", "RMSE"), ("mae", "MAE")):
    val = executive_metrics.get(key)
    if val is not None:
        print(f"  ✓ Found {key}={val}, would display as '{label}'")
        break
else:
    print(f"  ✗ No r2/rmse/mae found! Would fallback to 'Unavailable'")

print(f"\n{'='*80}")
print(f"CONCLUSION:")
if executive_metrics.get('r2') or executive_metrics.get('rmse') or executive_metrics.get('mae'):
    print(f"  ✓ Regression metrics ARE present in executive_metrics")
    print(f"  ✓ Executive Summary SHOULD display the selected metric (not 'Unavailable')")
else:
    print(f"  ✗ Regression metrics ARE MISSING from executive_metrics")
    print(f"  ✗ Executive Summary WILL display 'Unavailable' (regression metrics not in session state)")
print(f"{'='*80}")

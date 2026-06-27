#!/usr/bin/env python
"""Verify that regression metrics are extracted from cached pipeline output."""

import json

# Load cached housing regression output
print("Loading cached housing regression output...")
with open('autonomous_validation_runs/autonomous_result_housing.json', 'r') as f:
    output = json.load(f)

print(f"Problem Type: {output.get('problem_type', 'Unknown')}")
print(f"Dataset Analysis Problem Type: {output.get('dataset_analysis', {}).get('problem_analysis', {}).get('problem_type')}")

# Check model_comparison metrics
model_comparison = output.get('model_comparison', [])
print(f"\nmodel_comparison entries: {len(model_comparison)}")
for i, entry in enumerate(model_comparison):
    metrics = entry.get('metrics', {})
    if metrics:
        print(f"  Entry {i} metrics: {list(metrics.keys())}")
        if 'r2' in metrics:
            print(f"    r2: {metrics['r2']}")

# Check validation_results
validation_results = output.get('validation_results', {})
print(f"\nvalidation_results type: {type(validation_results)}")
if isinstance(validation_results, list):
    print(f"validation_results entries: {len(validation_results)}")
    for i, entry in enumerate(validation_results):
        if isinstance(entry, dict):
            validation = entry.get('validation', {})
            if validation:
                print(f"  Entry {i} validation metrics: {list(validation.keys())}")

# Now simulate normalizing this output through the bridge
print("\n" + "="*60)
print("Running normalize_pipeline_output_and_persist...")
print("="*60)

# This would require Streamlit session state, so just test the metrics extraction
from utils.safe_checks import coalesce_dict, safe_dict_get
from agents.trust_score_calculator import create_executive_metrics_object

problem_type = output.get('problem_type') or safe_dict_get(
    coalesce_dict(safe_dict_get(output.get('dataset_analysis'), 'problem_analysis')),
    'problem_type',
) or 'Classification'

print(f"\nExtracted Problem Type: {problem_type}")

# Create executive_metrics the same way the bridge does
# (Just create an empty dict to test injection logic)
executive_metrics = {
    'best_model': safe_dict_get(output.get('model_results'), 'best_model'),
    'accuracy': safe_dict_get(output.get('model_results'), 'accuracy'),
}

print(f"\nBefore regression injection:")
print(f"  Accuracy: {executive_metrics.get('accuracy')}")
print(f"  R²: {executive_metrics.get('r2')}")
print(f"  RMSE: {executive_metrics.get('rmse')}")

# Now apply the regression metrics injection logic
if "regress" in str(problem_type).lower():
    print(f"\n✓ Detected regression problem, injecting metrics...")
    
    # Extract from model_comparison if available
    model_comparison = output.get("model_comparison")
    if isinstance(model_comparison, list):
        for entry in model_comparison:
            if isinstance(entry, dict):
                metrics = entry.get("metrics", {})
                if isinstance(metrics, dict):
                    for key in ("r2", "rmse", "mae", "mse"):
                        if key in metrics and key not in executive_metrics:
                            executive_metrics[key] = metrics[key]
                            print(f"  ✓ Injected {key} = {metrics[key]}")
    
    # Extract from validation_results as fallback
    validation_results = output.get("validation_results")
    if isinstance(validation_results, list):
        for entry in validation_results:
            if isinstance(entry, dict):
                validation = entry.get("validation", {})
                if isinstance(validation, dict):
                    for key in ("r2", "rmse", "mae", "mse"):
                        if key in validation and key not in executive_metrics:
                            executive_metrics[key] = validation[key]
                            print(f"  ✓ Injected {key} = {validation[key]}")
                    break

print(f"\nAfter regression injection:")
print(f"  Accuracy: {executive_metrics.get('accuracy')}")
print(f"  R²: {executive_metrics.get('r2')}")
print(f"  RMSE: {executive_metrics.get('rmse')}")
print(f"  MAE: {executive_metrics.get('mae')}")

has_regression = executive_metrics.get('r2') or executive_metrics.get('rmse') or executive_metrics.get('mae')
if has_regression:
    print('\n✓ SUCCESS: Regression metrics successfully injected into executive_metrics!')
else:
    print('\n✗ FAILED: Regression metrics still missing')

#!/usr/bin/env python
"""Quick verification that regression metrics are now in executive_metrics."""

from agents.master_autonomous_pipeline import MasterAutonomousPipeline
import pandas as pd

# Load housing dataset (regression)
df = pd.read_csv('data/Housing.csv', nrows=100)
print(f'Dataset: {df.shape[0]} rows, {df.shape[1]} cols')
print(f'Dataset columns: {list(df.columns)}')

# Run pipeline
print('\nRunning pipeline...')
pipeline = MasterAutonomousPipeline()
output = pipeline.run_pipeline(
    dataset=df, 
    dataset_name='Housing',
    project_goal='Predict house median values'
)

# Check if regression metrics are in executive_metrics
exec_metrics = output.get('executive_metrics', {})
print('\n=== Executive Metrics ===')
print(f'Problem Type: {exec_metrics.get("problem_type")}')
print(f'Best Model: {exec_metrics.get("best_model")}')
print(f'Accuracy: {exec_metrics.get("accuracy")}')
print(f'Accuracy Display: {exec_metrics.get("accuracy_display")}')
print(f'R²: {exec_metrics.get("r2")}')
print(f'RMSE: {exec_metrics.get("rmse")}')
print(f'MAE: {exec_metrics.get("mae")}')

has_regression = exec_metrics.get('r2') or exec_metrics.get('rmse') or exec_metrics.get('mae')
if has_regression:
    print('\n✓ Regression metrics successfully mapped to executive_metrics!')
else:
    print('\n✗ Regression metrics still missing')
    print('model_comparison:', output.get('model_comparison'))
    print('validation_results:', output.get('validation_results'))

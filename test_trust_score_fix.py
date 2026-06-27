#!/usr/bin/env python3
"""Quick validation test for trust_score fix"""

import sys
import os
sys.path.insert(0, os.getcwd())

from agents.master_autonomous_pipeline import MasterAutonomousPipeline
import pandas as pd
import json

# Create pipeline instance
pipeline = MasterAutonomousPipeline()

# Load test dataset
df = pd.read_csv('data/Titanic-Dataset.csv')

print("=" * 50)
print("Testing trust_score fix...")
print("=" * 50)

try:
    result = pipeline.run_pipeline(
        dataset=df,
        project_goal='Predict passenger survival on Titanic',
        target_column='Survived',
        use_cache=False
    )
    
    # Check key fields
    print("\n=== TRUST_SCORE CHECK ===")
    trust_score_raw = result.get('ai_trust_results', {}).get('trust_score')
    print(f"trust_score value: {trust_score_raw}")
    print(f"trust_score type: {type(trust_score_raw).__name__}")
    print(f"Is numeric: {isinstance(trust_score_raw, (int, float))}")
    
    print("\n=== CONFIDENCE_VALUE CHECK ===")
    confidence_val = result.get('final_ai_confidence_score')
    print(f"confidence_value: {confidence_val}")
    print(f"confidence_value type: {type(confidence_val).__name__}")
    print(f"Is numeric: {isinstance(confidence_val, (int, float))}")
    
    print("\n=== MODEL VERSION CHECK ===")
    model_version = result.get('final_report_payload', {}).get('model_version')
    print(f"model_version: {model_version}")
    print(f"model_version type: {type(model_version).__name__}")
    
    print("\n=== PRODUCTION_MODEL CHECK ===")
    production_model = result.get('final_report_payload', {}).get('production_model')
    print(f"production_model: {production_model}")
    
    print("\n=== FINAL_SCORES CHECK ===")
    final_scores = result.get('final_scores', {})
    print(f"final_scores: {json.dumps(final_scores, indent=2, default=str)}")
    
    # Validation
    errors = []
    if not isinstance(trust_score_raw, (int, float)):
        errors.append(f"❌ trust_score is not numeric: {type(trust_score_raw).__name__}")
    else:
        print(f"✓ trust_score is numeric")
        
    if not isinstance(confidence_val, (int, float)):
        errors.append(f"❌ confidence_value is not numeric: {type(confidence_val).__name__}")
    else:
        print(f"✓ confidence_value is numeric")
        
    if model_version is None or model_version == "":
        errors.append(f"❌ model_version is empty")
    else:
        print(f"✓ model_version is set: {model_version}")
    
    if errors:
        print("\n" + "=" * 50)
        print("VALIDATION FAILED:")
        for err in errors:
            print(err)
        print("=" * 50)
        sys.exit(1)
    else:
        print("\n" + "=" * 50)
        print("✓ ALL VALIDATIONS PASSED")
        print("=" * 50)
        sys.exit(0)
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

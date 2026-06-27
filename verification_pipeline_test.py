#!/usr/bin/env python3
"""Complete pipeline verification with Titanic dataset"""

import sys
import os
sys.path.insert(0, os.getcwd())

import json
import pandas as pd
from agents.master_autonomous_pipeline import MasterAutonomousPipeline

print("=" * 80)
print("AUTODS-AGENT COMPLETE PIPELINE VERIFICATION")
print("Dataset: Titanic")
print("=" * 80)

pipeline = MasterAutonomousPipeline()
df = pd.read_csv('data/Titanic-Dataset.csv')

print(f"\n[STEP 1] Loading dataset: {df.shape[0]} rows, {df.shape[1]} columns")

try:
    print("\n[STEP 2] Running full pipeline...")
    result = pipeline.run_pipeline(
        dataset=df,
        project_goal='Predict passenger survival on Titanic',
        target_column='Survived',
        use_cache=False
    )
    
    print("\n" + "=" * 80)
    print("CAPTURED METRIC VALUES")
    print("=" * 80)
    
    # Extract key values
    metrics = {}
    
    # Dataset Quality Score
    dataset_score = result.get('final_report_payload', {}).get('health_score')
    metrics['dataset_quality_score'] = dataset_score
    print(f"\n1. Dataset Quality Score: {dataset_score}")
    
    # Confidence Level / Reliability Score
    confidence = result.get('final_ai_confidence_score')
    metrics['confidence_level'] = confidence
    print(f"2. Confidence Level: {confidence}")
    
    # Production Readiness
    deployment = result.get('deployment_readiness', {})
    readiness_score = deployment.get('readiness_score')
    metrics['production_readiness'] = readiness_score
    print(f"3. Production Readiness: {readiness_score}")
    
    # Recommended Model
    model_name = result.get('final_report_payload', {}).get('best_model')
    metrics['recommended_model'] = model_name
    print(f"4. Recommended Model: {model_name}")
    
    # Model Version
    model_version = result.get('final_report_payload', {}).get('model_version')
    metrics['model_version'] = model_version
    print(f"5. Model Version: {model_version}")
    
    # Risk Level
    risk_level = deployment.get('risk_level')
    metrics['risk_level'] = risk_level
    print(f"6. Risk Level: {risk_level}")
    
    # Trust Score
    trust_score = result.get('final_report_payload', {}).get('trust_score')
    metrics['trust_score'] = trust_score
    print(f"7. Trust Score: {trust_score}")
    
    # Final Score
    final_score = result.get('final_report_payload', {}).get('final_score')
    metrics['final_score'] = final_score
    print(f"8. Final Score: {final_score}")
    
    # Health Grade
    health_grade = result.get('final_report_payload', {}).get('health_grade')
    metrics['health_grade'] = health_grade
    print(f"9. Health Grade: {health_grade}")
    
    # Problem Type
    problem_type = result.get('final_report_payload', {}).get('problem_type')
    metrics['problem_type'] = problem_type
    print(f"10. Problem Type: {problem_type}")
    
    print("\n" + "=" * 80)
    print("CHIEF DECISION PANEL DATA")
    print("=" * 80)
    
    # Get chief decision data if available
    chief_data = result.get('chief_decision', {})
    print(f"\nChief Decision Data Keys: {list(chief_data.keys())}")
    print(f"Chief Pipeline Complete: {chief_data.get('pipeline_complete')}")
    print(f"Chief Model Name: {chief_data.get('model_name')}")
    print(f"Chief Confidence: {chief_data.get('confidence')}")
    print(f"Chief Risk Level: {chief_data.get('risk_level')}")
    print(f"Chief Deployment Label: {chief_data.get('deployment_label')}")
    print(f"Chief Trust Score: {chief_data.get('trust_score')}")
    
    print("\n" + "=" * 80)
    print("FINAL REPORT PAYLOAD STRUCTURE")
    print("=" * 80)
    
    payload = result.get('final_report_payload', {})
    print(f"\nPayload Keys ({len(payload.keys())}): {list(payload.keys())}")
    
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    
    errors = []
    
    # Check all metrics are numeric
    if not isinstance(trust_score, (int, float)):
        errors.append(f"❌ trust_score not numeric: {type(trust_score).__name__} = {trust_score}")
    else:
        print(f"✓ trust_score is numeric: {trust_score}")
    
    if not isinstance(confidence, (int, float)):
        errors.append(f"❌ confidence_level not numeric: {type(confidence).__name__}")
    else:
        print(f"✓ confidence_level is numeric: {confidence}")
    
    if not isinstance(dataset_score, (int, float)):
        errors.append(f"❌ dataset_quality_score not numeric: {type(dataset_score).__name__}")
    else:
        print(f"✓ dataset_quality_score is numeric: {dataset_score}")
    
    if model_version is None or model_version == "":
        errors.append(f"❌ model_version is empty")
    else:
        print(f"✓ model_version is set: {model_version}")
    
    if model_name is None or model_name == "":
        errors.append(f"❌ recommended_model is empty")
    else:
        print(f"✓ recommended_model is set: {model_name}")
    
    if not isinstance(readiness_score, (int, float)):
        errors.append(f"❌ production_readiness not numeric")
    else:
        print(f"✓ production_readiness is numeric: {readiness_score}")
    
    # Save metrics to JSON for UI verification
    metrics_file = 'verification_pipeline_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n✓ Metrics saved to {metrics_file}")
    
    print("\n" + "=" * 80)
    if errors:
        print("VALIDATION FAILURES:")
        for err in errors:
            print(err)
        sys.exit(1)
    else:
        print("✓ ALL VALIDATIONS PASSED")
        print("=" * 80)
        sys.exit(0)
    
except Exception as e:
    print(f"\n❌ PIPELINE ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

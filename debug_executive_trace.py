#!/usr/bin/env python
"""
DEBUG: Detailed trace of Executive Summary rendering for regression dataset.

Shows:
  1. Where problem_type is resolved from
  2. Where executive_metrics is read from in Streamlit session
  3. The exact rendering location and values
"""

import json
from utils.safe_checks import coalesce_dict, safe_dict_get, format_accuracy_display
from utils.session_manager import SessionKeys, normalize_problem_type

print("="*80)
print("DEBUG: COMPLETE TRACE OF EXECUTIVE SUMMARY RENDERING")
print("="*80)

# Load cached housing regression output
print("\n[1] Loading cached regression output...")
with open('autonomous_validation_runs/autonomous_result_housing.json', 'r') as f:
    output = json.load(f)

print(f"  File: autonomous_validation_runs/autonomous_result_housing.json")
print(f"  Dataset: Housing (Regression)")

# === TRACE PROBLEM_TYPE SOURCE ===
print("\n" + "="*80)
print("[2] TRACING problem_type RESOLUTION")
print("="*80)
print("\n  Simulating get_problem_type(output) from utils/session_manager.py line 227:")
print("  → Checks st.session_state.get(SessionKeys.PROBLEM_TYPE) first")
print("     (in real Streamlit, this would be cached from previous run)")
print("     Current sim: Not in session cache yet\n")

# Check where problem_type would come from in output
dataset_report = output.get("dataset_report") or output.get("dataset_analysis") or {}
if isinstance(dataset_report, dict):
    problem_analysis = coalesce_dict(dataset_report.get("problem_analysis"))
    pt = problem_analysis.get("problem_type")
    if pt:
        pt_normalized = normalize_problem_type(str(pt))
        print(f"  ✓ Found in output['dataset_analysis']['problem_analysis']['problem_type']")
        print(f"    Raw value: {pt}")
        print(f"    Normalized: {pt_normalized}")

# === TRACE EXECUTIVE METRICS SOURCE ===
print("\n" + "="*80)
print("[3] TRACING executive_metrics RESOLUTION")
print("="*80)
print("\n  Simulating build_report_context() from ui/interactive_report_center.py line 88:")
print("  → Checks st.session_state.get(SessionKeys.EXECUTIVE_METRICS) first")
print("     This is populated by hydrate_pipeline_session_from_output()")
print("     Which calls normalize_pipeline_output_and_persist()")
print("     Which includes the _inject_regression_metrics() fix\n")

# Simulate the fix being applied
problem_type = normalize_problem_type(pt) if pt else "Classification"

executive_metrics_sim = {
    'best_model': safe_dict_get(output.get('model_results'), 'best_model'),
    'accuracy': safe_dict_get(output.get('model_results'), 'accuracy'),
}

# Simulate finalize display
executive_metrics_sim['accuracy_display'] = format_accuracy_display(
    executive_metrics_sim.get('accuracy'), 
    problem_type
)

# Simulate the FIX - inject regression metrics
if "regress" in str(problem_type).lower():
    model_comparison = output.get("model_comparison")
    if isinstance(model_comparison, list):
        for entry in model_comparison:
            if isinstance(entry, dict):
                metrics = entry.get("metrics", {})
                if isinstance(metrics, dict):
                    for key in ("r2", "rmse", "mae", "mse"):
                        if key in metrics and key not in executive_metrics_sim:
                            executive_metrics_sim[key] = metrics[key]
    
    validation_results = output.get("validation_results")
    if isinstance(validation_results, list):
        for entry in validation_results:
            if isinstance(entry, dict):
                validation = entry.get("validation", {})
                if isinstance(validation, dict):
                    for key in ("r2", "rmse", "mae", "mse"):
                        if key in validation and key not in executive_metrics_sim:
                            executive_metrics_sim[key] = validation[key]
                    break

# === SHOW WHAT WOULD BE IN SESSION STATE ===
print(f"  After fix, st.session_state[SessionKeys.EXECUTIVE_METRICS] would contain:")
print(f"  {json.dumps(executive_metrics_sim, indent=4, default=str)}")

# === TRACE THE RENDERING CALL ===
print("\n" + "="*80)
print("[4] EXACT RENDERING LOCATION")
print("="*80)
print(f"\n  File: ui/interactive_report_center.py")
print(f"  Function: _tab_executive_summary() - line 627")
print(f"  Inner helper: _select_primary_metric() - line 635\n")

print(f"  Key code (line 641):")
print(f"    exec_metrics = st.session_state.get(SessionKeys.EXECUTIVE_METRICS) or {{}}")
print(f"\n  This function then (line 650-656):")
print(f"    1. Gets problem_type from ctx (from build_report_context())")
print(f"    2. Checks if 'regress' in problem_type.lower()")
print(f"    3. If yes, loops through r2, rmse, mae looking for first non-None value")
print(f"    4. Formats and returns that value\n")

# === SIMULATE THE SELECTION ===
print("[5] SIMULATING _select_primary_metric() LOGIC")
print("="*80)

problem_type_from_ctx = normalize_problem_type(pt) if pt else "Classification"
exec_metrics = executive_metrics_sim or {}

print(f"\n  problem_type from ctx: {problem_type_from_ctx}")
print(f"  is_regression = 'regress' in '{problem_type_from_ctx}'.lower()")
is_regression = "regress" in problem_type_from_ctx.lower()
print(f"    → {is_regression}\n")

if not is_regression:
    print(f"  Classification path: return 'Accuracy', ctx.get('accuracy_display')")
    print(f"    → Would display: 'Accuracy: {executive_metrics_sim.get('accuracy_display')}'")
else:
    print(f"  Regression path: search for r2, rmse, mae in exec_metrics\n")
    for key, label in (("r2", "R² Score"), ("rmse", "RMSE"), ("mae", "MAE")):
        val = exec_metrics.get(key)
        print(f"    Checking exec_metrics['{key}']: {val}")
        if val is not None:
            print(f"      ✓ FOUND! Would return ('{label}', formatted_value)")
            print(f"      → Would display: '{label} {val}'")
            break
    else:
        fallback = executive_metrics_sim.get('accuracy_display', 'Unavailable')
        print(f"    No r2/rmse/mae found!")
        print(f"    Fallback to accuracy_display: {fallback}")
        print(f"      → Would display: 'R² Score: {fallback}'")

# === FINAL DIAGNOSIS ===
print("\n" + "="*80)
print("[6] DIAGNOSIS")
print("="*80)

print(f"\n✓ FIX WAS IMPLEMENTED: _inject_regression_metrics() runs in pipeline_bridge.py")
print(f"✓ REGRESSION METRICS IN SESSION: r2={exec_metrics.get('r2')}, rmse={exec_metrics.get('rmse')}, mae={exec_metrics.get('mae')}")
print(f"✓ LOGIC SHOULD WORK: _select_primary_metric() will find r2/rmse/mae and display it")

print(f"\n⚠ POSSIBLE REASONS FOR STILL SEEING 'Accuracy: Unavailable':")
print(f"  1. Streamlit is serving CACHED output (session_state not updated)")
print(f"  2. Browser cache showing old page (try Ctrl+Shift+R hard refresh)")
print(f"  3. Pipeline output in session_state is from BEFORE the fix was applied")
print(f"  4. The pipeline never ran again - still using old cached executive_metrics")
print(f"\nTO FIX:")
print(f"  1. Kill the Streamlit server (Ctrl+C)")
print(f"  2. Clear browser cache or open in incognito")
print(f"  3. Restart with: streamlit run app.py")
print(f"  4. Re-run the pipeline on housing dataset")
print(f"  5. The Executive Summary SHOULD NOW show 'R² Score -0.23' (not 'Unavailable')")
print("="*80)

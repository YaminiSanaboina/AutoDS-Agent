#!/usr/bin/env python3
"""Quick verification of code changes without running full pipeline"""

import sys
import os
sys.path.insert(0, os.getcwd())

print("=" * 80)
print("AUTODS VERIFICATION - CODE CHANGES CHECK")
print("=" * 80)

# Check 1: Verify confidence_value is computed in master_autonomous_pipeline.py
print("\n[CHECK 1] Verifying confidence_value computation...")
with open('agents/master_autonomous_pipeline.py', 'r') as f:
    content = f.read()
    if 'confidence_value = float(trust_score)' in content:
        print("✓ confidence_value computation found")
    else:
        print("❌ confidence_value computation NOT found")
        sys.exit(1)

# Check 2: Verify trust_score is forced to numeric
print("[CHECK 2] Verifying trust_score numeric conversion...")
if 'ts = float(ts) if isinstance(ts, (int, float)) else 0' in content:
    print("✓ trust_score numeric conversion found")
else:
    print("❌ trust_score numeric conversion NOT found")
    sys.exit(1)

# Check 3: Verify model_version uses registry
print("[CHECK 3] Verifying model_version from registry...")
if 'model_registry_entry.get("version")' in content:
    print("✓ model_version from registry found")
else:
    print("❌ model_version from registry NOT found")
    sys.exit(1)

# Check 4: Verify production_model in final_report_payload
print("[CHECK 4] Verifying production_model in payload...")
if '"production_model": production_model' in content:
    print("✓ production_model in payload found")
else:
    print("❌ production_model in payload NOT found")
    sys.exit(1)

# Check 5: Verify final_ai_confidence_score set to confidence_value
print("[CHECK 5] Verifying final_ai_confidence_score...")
if '"final_ai_confidence_score": confidence_value' in content or 'final_ai_confidence_score": confidence_value' in content:
    print("✓ final_ai_confidence_score = confidence_value found")
else:
    print("❌ final_ai_confidence_score = confidence_value NOT found")
    sys.exit(1)

# Check 6: Verify trust_score in final_report_payload
print("[CHECK 6] Verifying trust_score in final_report_payload...")
if '"trust_score": ai_trust_results.get("trust_score", 0)' in content:
    print("✓ trust_score in final_report_payload found")
else:
    print("❌ trust_score in final_report_payload NOT found")
    sys.exit(1)

# Check 7: Search for old label names that should be updated
print("\n[CHECK 7] Searching for old label names in UI...")
old_labels = {
    'Trust Score': ['ui/dashboard.py', 'ui/interactive_report_center.py', 'ui/chief_decision_panel.py'],
    'Health Score': ['ui/dashboard.py', 'ui/chief_decision_panel.py'],
    'Deployment Readiness': ['ui/dashboard.py', 'ui/chief_decision_panel.py'],
    'Executive Summary': ['ui/interactive_report_center.py', 'ui/dashboard.py'],
    'Dataset Analysis': ['ui/interactive_report_center.py'],
    'Models': ['ui/interactive_report_center.py'],
    'Explainability': ['ui/interactive_report_center.py'],
    'Trust & Risk': ['ui/interactive_report_center.py'],
    'Final Decision': ['ui/interactive_report_center.py'],
}

old_label_occurrences = {}
for label, files in old_labels.items():
    for filepath in files:
        full_path = os.path.join('.', filepath)
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    count = 0
                    for i, line in enumerate(lines):
                        if label in line and not line.strip().startswith('#'):
                            if label not in old_label_occurrences:
                                old_label_occurrences[label] = []
                            count += 1
                            old_label_occurrences[label].append((filepath, i + 1, line.strip()[:80]))
                    if count > 0:
                        print(f"  Found '{label}' {count} times in {filepath}")
            except Exception as e:
                pass  # Skip files that can't be read

if old_label_occurrences:
    print(f"\n⚠ Old label names still present: {len(old_label_occurrences)} labels found")
else:
    print("✓ No old label names found (or already updated)")

# Check 8: Search for raw JSON output in UI
print("\n[CHECK 8] Searching for raw JSON in UI...")
json_patterns = ['st.json(', 'json.dumps(', 'str(metrics)', 'str(validation)']
json_found = False
ui_files = []
for root, dirs, files in os.walk('ui'):
    for f in files:
        if f.endswith('.py'):
            ui_files.append(os.path.join(root, f))

for filepath in ui_files:
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if any(pattern in line for pattern in json_patterns) and not line.strip().startswith('#'):
                    if 'legacy_pages' not in filepath:  # Legacy pages are OK
                        json_found = True
                        print(f"  {filepath}:{i+1} - {line.strip()[:80]}")
    except Exception as e:
        pass  # Skip files that can't be read

if not json_found:
    print("✓ No raw JSON found in main UI (legacy pages OK)")

print("\n" + "=" * 80)
print("CODE CHANGES VERIFICATION SUMMARY")
print("=" * 80)
print("✓ All 6 critical code changes verified")
if old_label_occurrences:
    print(f"⚠ {len(old_label_occurrences)} old label names still present (non-critical)")
if json_found:
    print("⚠ Raw JSON found in UI (non-critical for core pipeline)")
print("\n✓ READY FOR PIPELINE EXECUTION TEST")
print("=" * 80)

sys.exit(0)

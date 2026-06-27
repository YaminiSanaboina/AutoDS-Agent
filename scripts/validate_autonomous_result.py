import json
import sys
from pathlib import Path

result_file = Path('autonomous_result_telco.json')
if not result_file.exists():
    print('ERROR: autonomous_result_telco.json not found')
    sys.exit(2)

with open(result_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

checks = {}

def non_empty(d, key):
    v = d.get(key)
    if v is None:
        return False, 'missing'
    if isinstance(v, (list, dict)) and len(v) == 0:
        return False, 'empty'
    if isinstance(v, str) and v.strip() == '':
        return False, 'blank'
    return True, ''

# Required top-level keys
required = [
    'dataset_analysis', 'cleaning_results', 'eda_results', 'feature_engineering_results',
    'model_results', 'model_versions', 'production_model', 'validation_history', 'improvement_history',
    'optimization_iterations', 'final_scores', 'final_ai_confidence_score', 'deployment_readiness',
    'final_report', 'stage_errors'
]

all_ok = True
for k in required:
    ok, reason = non_empty(data, k)
    checks[k] = {'ok': ok, 'reason': reason}
    if not ok:
        all_ok = False

# Additional numeric checks
fs = data.get('final_scores', {})
if isinstance(fs, dict):
    checks['final_scores_values'] = {}
    for metric in ['dataset_score','model_score','trust_score','mlops_score','overall_score']:
        v = fs.get(metric)
        checks['final_scores_values'][metric] = {'value': v, 'ok': (v is not None and not (isinstance(v, (int,float)) and v==0))}
        if not checks['final_scores_values'][metric]['ok']:
            all_ok = False

# Check artifacts exist
artifact_ok = True
artifacts = []
pp = Path('deployment_package')
if pp.exists():
    for dep in pp.glob('DEP_*'):
        v1 = dep / 'v1'
        if v1.exists():
            for fname in ['best_model.pkl','target_encoder.pkl','preprocessing_pipeline.pkl','feature_schema.json','model_metadata.json']:
                p = v1 / fname
                artifacts.append(str(p))
                if not p.exists():
                    artifact_ok = False
else:
    artifact_ok = False

checks['artifacts'] = {'found': artifact_ok, 'paths': artifacts}
if not artifact_ok:
    all_ok = False

# Check final report artifact
final_report = data.get('final_report')
report_ok = False
report_path = None
if isinstance(final_report, dict):
    report_path = final_report.get('path')
    report_ok = bool(report_path and Path(report_path).exists())
checks['final_report'] = {'ok': report_ok, 'path': report_path}
if not report_ok:
    all_ok = False

# Check stage error list exists
stage_errors = data.get('stage_errors')
checks['stage_errors'] = {'ok': isinstance(stage_errors, list), 'count': len(stage_errors) if isinstance(stage_errors, list) else 0}
if not isinstance(stage_errors, list):
    all_ok = False

report = {'all_ok': all_ok, 'checks': checks}
with open('autonomous_validation_report_telco.json','w',encoding='utf-8') as f:
    json.dump(report, f, indent=2)

print('Validation completed. all_ok=', all_ok)
print('Report written to autonomous_validation_report_telco.json')
if not all_ok:
    sys.exit(1)

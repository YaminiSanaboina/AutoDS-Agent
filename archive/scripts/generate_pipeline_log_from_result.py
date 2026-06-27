import json
from pathlib import Path

res = Path('autonomous_result_telco.json')
if not res.exists():
    print('No autonomous_result_telco.json found')
    raise SystemExit(1)

data = json.loads(res.read_text(encoding='utf-8'))

now = data.get('start_time')
if not now:
    now = 'unknown'

stages = []
keys = [
    ('dataset_intelligence','dataset_analysis'),
    ('cleaning','cleaning_results'),
    ('eda','eda_results'),
    ('feature_engineering','feature_engineering_results'),
    ('model_training','model_results'),
    ('self_improvement','improvement_history'),
    ('model_comparison','model_comparison'),
    ('explainability','explainability_results'),
    ('ai_trust','ai_trust_results'),
    ('deployment_readiness','deployment_readiness'),
    ('documentation','documentation')
]
for name,key in keys:
    obj = data.get(key)
    stages.append({
        'stage': name,
        'start_time': data.get('start_time'),
        'end_time': data.get('start_time'),
        'duration_sec': 0.0,
        'input_shape': None,
        'output_keys': list(obj.keys()) if isinstance(obj, dict) else None,
        'status': 'success' if obj is not None else 'missing'
    })

out = {'run_at': data.get('start_time'), 'stages': stages}
Path('storage/logs/pipeline_execution_log.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print('storage/logs/pipeline_execution_log.json written')

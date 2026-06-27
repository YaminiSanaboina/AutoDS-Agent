import sys
import time
import json
from pathlib import Path
import sys
from pathlib import Path
workspace_root = Path(__file__).resolve().parents[1]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

import pandas as pd
from agents.master_autonomous_pipeline import MasterAutonomousPipeline

if len(sys.argv) < 3:
    print('Usage: run_single_validation.py <csv_path> <name>')
    sys.exit(2)

csv_path = sys.argv[1]
name = sys.argv[2]

df = pd.read_csv(csv_path)
pipeline = MasterAutonomousPipeline()
start = time.time()
try:
    output = pipeline.run_pipeline(dataset=df, dataset_name=name, project_goal=f"Validate {name}", smart_mode=True, max_seconds=300)
    duration = time.time() - start
    out_dir = Path('autonomous_validation_runs')
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"autonomous_result_{name.lower()}.json", 'w', encoding='utf-8') as fh:
        json.dump(output, fh, default=str, indent=2)
    print(f"Completed {name} in {duration:.1f}s")
except Exception as e:
    duration = time.time() - start
    print(f"Failed {name} after {duration:.1f}s: {e}")
    sys.exit(1)

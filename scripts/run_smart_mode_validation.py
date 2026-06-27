import time
import json
from pathlib import Path
import sys
import os
import pandas as pd

# Ensure workspace root is on sys.path so package imports work
workspace_root = Path(__file__).resolve().parents[1]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from agents.master_autonomous_pipeline import MasterAutonomousPipeline

DATA = [
    ("data/WA_Fn-UseC_-Telco-Customer-Churn.csv", "Telco"),
    ("data/Iris.csv", "Iris"),
    ("data/Housing.csv", "Housing"),
]

out_dir = Path("autonomous_validation_runs")
out_dir.mkdir(exist_ok=True)

results = {}

for path, name in DATA:
    print(f"Running pipeline for {name} -> {path}")
    df = pd.read_csv(path)
    pipeline = MasterAutonomousPipeline()
    start = time.time()
    try:
        output = pipeline.run_pipeline(dataset=df, dataset_name=name, project_goal=f"Validate {name}", smart_mode=True, max_seconds=300)
        duration = time.time() - start
        stage_durations = output.get('stage_durations', {})
        total_runtime = output.get('total_runtime', duration)
        results[name] = {
            "duration": duration,
            "success": True,
            "final_score": output.get('final_score'),
            "notes": output.get('notes', []),
            "stage_durations": stage_durations,
            "total_runtime": total_runtime,
        }
        with open(out_dir / f"autonomous_result_{name.lower()}.json", "w", encoding="utf-8") as fh:
            json.dump(output, fh, default=str, indent=2)
        print(f"Completed {name} in {duration:.1f}s - score: {output.get('final_score')}")
    except Exception as e:
        duration = time.time() - start
        results[name] = {"duration": duration, "success": False, "error": str(e)}
        print(f"Failed {name} after {duration:.1f}s: {e}")

with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)

print("All runs finished. Summary:")
print(json.dumps(results, indent=2))

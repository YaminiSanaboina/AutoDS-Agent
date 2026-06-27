import pandas as pd
import json
import sys
import os
from pathlib import Path

# Ensure workspace root is on sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.master_autonomous_pipeline import MasterAutonomousPipeline

def main(argv=None):
    argv = sys.argv if argv is None else argv
    if len(argv) < 3:
        print('Usage: run_pipeline_dataset.py <csv_path> <output_name>')
        raise SystemExit(2)
    csv_path = argv[1]
    out_name = argv[2]
    df = pd.read_csv(csv_path)
    mpipeline = MasterAutonomousPipeline()
    res = mpipeline.run_pipeline(df, dataset_name=Path(csv_path).stem, project_goal=f'Validate dataset {out_name}')
    with open(out_name, 'w', encoding='utf-8') as f:
        json.dump(res, f, default=str, indent=2)
    print('Wrote', out_name)


if __name__ == '__main__':
    main()

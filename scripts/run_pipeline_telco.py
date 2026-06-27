import json
import sys
import os
import pandas as pd

# Ensure workspace root is on sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.master_autonomous_pipeline import MasterAutonomousPipeline


def main():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    pipeline = MasterAutonomousPipeline()
    result = pipeline.run_pipeline(df, dataset_name="Telco-Churn", project_goal="Validate self-improvement agent")
    print("Pipeline returned keys:", list(result.keys()))
    with open("autonomous_result_telco.json", "w", encoding="utf-8") as f:
        json.dump(result, f, default=str, indent=2)


if __name__ == "__main__":
    main()

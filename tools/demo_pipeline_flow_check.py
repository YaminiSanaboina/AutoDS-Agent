#!/usr/bin/env python
"""Verify end-to-end demo flow for Iris and Titanic datasets."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from agents.master_autonomous_pipeline import MasterAutonomousPipeline
from utils.pipeline_bridge import normalize_pipeline_output

DATASETS = {
    "Iris": "data/Iris.csv",
    "Titanic": "data/Titanic-Dataset.csv",
}

KEYS = [
    "best_model",
    "final_score",
    "trust_score",
    "deployment_readiness",
    "final_report",
    "recommendation",
]


def audit_dataset(name, path):
    print(f"\n=== AUDIT {name} ===")
    df = pd.read_csv(path)
    print(f"Loaded dataset {name}: {df.shape}")

    pipeline = MasterAutonomousPipeline()
    output = pipeline.run_pipeline(
        dataset=df,
        dataset_name=name,
        project_goal=f"Demo audit for {name}",
        constraints={},
        smart_mode=True,
        max_seconds=300,
        progress_callback=lambda evt: None,
        use_cache=False,
    )

    normalized = normalize_pipeline_output(output)
    result = {
        "dataset": name,
        "output_keys": {},
        "normalized_keys": {},
        "best_model": None,
        "final_score": None,
        "trust_score": None,
        "deployment_risk": None,
        "final_report_path": None,
        "final_report_payload": None,
    }

    model_results = normalized.get("model_results") or {}
    trust = normalized.get("ai_trust_results") or {}
    deploy = normalized.get("deployment_readiness") or {}
    final_report = normalized.get("final_report") or {}

    result["best_model"] = model_results.get("best_model")
    result["final_score"] = normalized.get("final_score") or normalized.get("final_scores", {}).get("overall_score")
    result["trust_score"] = trust.get("trust_score")
    result["deployment_risk"] = deploy.get("risk_level") or deploy.get("risk")
    result["final_report_path"] = final_report.get("path")
    result["final_report_payload"] = bool(final_report.get("payload"))

    for key in KEYS:
        result["output_keys"][key] = {
            "present": key in output,
            "value": str(output.get(key))[:80] if output.get(key) is not None else None,
        }
    for key in ["dataset_report", "model_results", "ai_trust_results", "deployment_readiness", "final_report"]:
        result["normalized_keys"][key] = key in normalized

    return result


def main():
    audit_results = []
    for name, path in DATASETS.items():
        if not Path(path).exists():
            print(f"ERROR: Missing dataset file {path}")
            continue
        audit_results.append(audit_dataset(name, path))

    out_path = Path("storage/logs/demo_pipeline_flow_check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\nAudit results saved to {out_path}")
    for result in audit_results:
        print(json.dumps(result, indent=2)[:1000])


if __name__ == "__main__":
    main()

"""Production stabilization verification across demo datasets."""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.master_autonomous_pipeline import MasterAutonomousPipeline
from ui.dashboard import build_dashboard_context
from ui.interactive_report_center import build_report_context
from utils.pipeline_bridge import apply_autonomous_result_to_session, normalize_pipeline_output
from utils.report_exports import build_excel_bytes, build_export_context_from_report_ctx
from utils.session_manager import SessionKeys, get_problem_type, init_session, normalize_problem_type
from utils.safe_checks import is_present

import streamlit as st

DATASETS = [
    ("Iris", ROOT / "data" / "Iris.csv", "Classification"),
    ("Titanic", ROOT / "data" / "Titanic-Dataset.csv", "Classification"),
    ("Heart Disease", ROOT / "data" / "Heart-Disease.csv", "Classification"),
    ("Housing", ROOT / "data" / "Housing.csv", "Regression"),
    ("Telco Churn", ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv", "Classification"),
]


def verify_dataset(name: str, csv_path: Path, expected_type: str) -> dict:
    result = {
        "dataset": name,
        "path": str(csv_path),
        "ok": False,
        "expected_problem_type": expected_type,
    }
    if not csv_path.exists():
        result["error"] = f"Missing dataset: {csv_path}"
        return result

    init_session()
    for key in (
        SessionKeys.PROBLEM_TYPE,
        SessionKeys.EXECUTIVE_METRICS,
        SessionKeys.AUTONOMOUS_RESULT,
        SessionKeys.REPORT_PAYLOAD,
    ):
        st.session_state.pop(key, None)

    try:
        df = pd.read_csv(csv_path)
        if name == "Telco Churn":
            df = df.drop(columns=["customerID"], errors="ignore")

        start = time.time()
        pipeline = MasterAutonomousPipeline()
        output = pipeline.run_pipeline(
            df,
            dataset_name=csv_path.stem,
            project_goal=f"Production verify — {name}",
            smart_mode=True,
            use_cache=False,
        )
        runtime = round(time.time() - start, 2)
        normalized = normalize_pipeline_output(output)
        apply_autonomous_result_to_session(normalized)

        problem_type = get_problem_type(normalized)
        model_results = normalized.get("model_results") or {}
        metrics = model_results.get("metrics") or {}
        executive = normalized.get("executive_metrics") or st.session_state.get(SessionKeys.EXECUTIVE_METRICS) or {}
        home_ctx = build_dashboard_context(normalized)
        report_ctx = build_report_context(normalized, df)
        export_ctx = build_export_context_from_report_ctx(report_ctx)
        pdf_payload = st.session_state.get(SessionKeys.REPORT_PAYLOAD) or {}

        home_acc = home_ctx.get("score_display")
        report_acc = report_ctx.get("accuracy_display")
        export_acc = export_ctx.get("accuracy_display")
        pdf_acc = pdf_payload.get("accuracy_display")

        result.update(
            {
                "runtime_seconds": runtime,
                "detected_problem_type": problem_type,
                "training_problem_type": normalize_problem_type(model_results.get("problem_type")),
                "best_model": model_results.get("best_model"),
                "accuracy_display": executive.get("accuracy_display") or home_acc,
                "trust_score": executive.get("trust_score") or (normalized.get("ai_trust_results") or {}).get("trust_score"),
                "models_trained": len(metrics),
                "model_names": sorted(metrics.keys()),
                "home_accuracy": home_acc,
                "report_accuracy": report_acc,
                "export_accuracy": export_acc,
                "pdf_accuracy": pdf_acc,
                "accuracy_consistent": len({home_acc, report_acc, export_acc, pdf_acc} - {None, "Unavailable"}) <= 1,
                "problem_type_ok": normalize_problem_type(problem_type) == normalize_problem_type(expected_type),
                "pdf_path": (normalized.get("final_report") or {}).get("path"),
                "excel_bytes": len(build_excel_bytes(export_ctx)),
            }
        )

        result["ok"] = (
            result["problem_type_ok"]
            and result["models_trained"] >= 5
            and is_present(result["best_model"])
            and is_present(result["accuracy_display"])
            and result["accuracy_display"] not in ("0.0%", "Unavailable")
            and result["accuracy_consistent"]
            and result["excel_bytes"] > 500
        )
    except Exception:
        result["error"] = traceback.format_exc()

    return result


def main() -> int:
    results = [verify_dataset(name, path, expected) for name, path, expected in DATASETS]
    report = {"all_ok": all(r.get("ok") for r in results), "datasets": results}
    out_path = ROOT / "reports" / "production_stabilization_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print("Production stabilization verification")
    print("=" * 60)
    for entry in results:
        status = "PASS" if entry.get("ok") else "FAIL"
        print(f"{status} — {entry['dataset']}")
        if entry.get("error"):
            print(entry["error"][:400])
        else:
            print(
                f"  runtime={entry.get('runtime_seconds')}s "
                f"type={entry.get('detected_problem_type')} "
                f"models={entry.get('models_trained')} "
                f"best={entry.get('best_model')} "
                f"acc={entry.get('accuracy_display')} "
                f"trust={entry.get('trust_score')}"
            )
    print(f"\nReport: {out_path}")
    print(f"Overall: {'PASS' if report['all_ok'] else 'FAIL'}")
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

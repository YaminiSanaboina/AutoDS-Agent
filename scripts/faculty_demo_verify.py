"""End-to-end faculty demo verification for Iris, Titanic, and Heart Disease."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.master_autonomous_pipeline import MasterAutonomousPipeline
from agents.report_agent import generate_pdf_report
from ui.interactive_report_center import build_report_context
from utils.pipeline_bridge import (
    build_stage_results_from_output,
    normalize_pipeline_output,
    validate_pipeline_output,
)
from utils.report_exports import build_excel_bytes, build_export_context_from_report_ctx

DATASETS = [
    ("Iris", ROOT / "data" / "Iris.csv"),
    ("Titanic", ROOT / "data" / "Titanic-Dataset.csv"),
    ("Heart Disease", ROOT / "data" / "Heart-Disease.csv"),
    ("Housing", ROOT / "data" / "Housing.csv"),
    ("Wine Quality", ROOT / "data" / "Wine-Quality.csv"),
]

REQUIRED_TOP_LEVEL = (
    "dataset_report",
    "cleaning_results",
    "eda_results",
    "model_results",
    "explainability_results",
    "ai_trust_results",
    "deployment_readiness",
    "final_report",
)


def _non_empty(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (dict, list)) and len(value) == 0:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def verify_dataset(name: str, csv_path: Path) -> dict:
    result = {"dataset": name, "path": str(csv_path), "ok": False, "checks": {}}
    if not csv_path.exists():
        result["error"] = f"Missing dataset file: {csv_path}"
        return result

    try:
        df = pd.read_csv(csv_path)
        pipeline = MasterAutonomousPipeline()
        output = pipeline.run_pipeline(
            df,
            dataset_name=csv_path.stem,
            project_goal=f"Faculty demo validation — {name}",
            smart_mode=True,
            use_cache=True,
        )
        normalized = normalize_pipeline_output(output)
        stage_results = build_stage_results_from_output(normalized)
        warnings = validate_pipeline_output(normalized)

        checks = result["checks"]
        checks["stage_count"] = len(stage_results)
        checks["stage_keys"] = sorted(stage_results.keys())
        checks["validation_warnings"] = warnings

        for key in REQUIRED_TOP_LEVEL:
            checks[key] = _non_empty(normalized.get(key))

        final_report = normalized.get("final_report") or {}
        payload = final_report.get("payload") if isinstance(final_report, dict) else None
        report_path = final_report.get("path") if isinstance(final_report, dict) else None
        checks["pdf_path_exists"] = bool(report_path and Path(report_path).exists())

        if payload and not checks["pdf_path_exists"]:
            try:
                regenerated = generate_pdf_report(payload)
                checks["pdf_regenerated"] = bool(regenerated and Path(regenerated).exists())
                checks["pdf_path_exists"] = checks["pdf_regenerated"]
            except Exception as exc:
                checks["pdf_regenerated"] = False
                checks["pdf_error"] = str(exc)

        ctx = build_report_context(normalized, df)
        export_ctx = build_export_context_from_report_ctx(ctx)
        try:
            excel_bytes = build_excel_bytes(export_ctx)
            checks["excel_bytes"] = len(excel_bytes) > 500
        except Exception as exc:
            checks["excel_bytes"] = False
            checks["excel_error"] = str(exc)

        stage_errors = normalized.get("stage_errors") or []
        checks["stage_errors_count"] = len(stage_errors) if isinstance(stage_errors, list) else -1

        critical_ok = (
            checks["stage_count"] >= 5
            and checks.get("dataset_report")
            and checks.get("model_results")
            and checks.get("final_report")
            and checks["pdf_path_exists"]
            and checks["excel_bytes"]
        )
        result["ok"] = critical_ok and not warnings
        if warnings and critical_ok:
            result["ok_with_warnings"] = True
            result["ok"] = True
    except Exception:
        result["error"] = traceback.format_exc()

    return result


def main() -> int:
    os.chdir(ROOT)
    results = [verify_dataset(name, path) for name, path in DATASETS]
    report = {
        "all_ok": all(r.get("ok") for r in results),
        "datasets": results,
    }
    out_path = ROOT / "reports" / "faculty_demo_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print("Faculty demo verification")
    print("=" * 40)
    for entry in results:
        status = "PASS" if entry.get("ok") else "FAIL"
        print(f"{status} — {entry['dataset']}")
        if entry.get("error"):
            print(entry["error"][:500])
        elif entry.get("checks"):
            checks = entry["checks"]
            print(
                f"  stages={checks.get('stage_count')} "
                f"pdf={checks.get('pdf_path_exists')} "
                f"excel={checks.get('excel_bytes')} "
                f"warnings={checks.get('validation_warnings')}"
            )
    print(f"\nReport: {out_path}")
    print(f"Overall: {'PASS' if report['all_ok'] else 'FAIL'}")
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""End-to-end headless validation for AutoDS Agent.

Runs the autonomous pipeline for provided datasets via AppTest and
collects diagnostics for FINAL_E2E_VALIDATION_REPORT.md.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Dict, Any

import pandas as pd

sys.path.insert(0, os.getcwd())
from streamlit.testing.v1.app_test import AppTest
from utils.session_manager import SessionKeys
from config import SMART_MODE_BUDGET_SECONDS


DATASETS = [
    ("Iris", os.path.join("data", "Iris.csv")),
    ("Telco", os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")),
    ("Housing", os.path.join("data", "Housing.csv")),
]


def run_dataset(name: str, path: str, timeout: float) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"dataset": name, "path": path, "status": "", "error": "", "runtime": None}
    if not os.path.exists(path):
        rec["status"] = "MISSING"
        rec["error"] = f"Dataset file not found: {path}"
        return rec

    df = pd.read_csv(path)

    at = AppTest.from_file("app.py", default_timeout=timeout)
    # AppTest uses a SafeSessionState that doesn't expose dict-like methods
    # (calling `st.session_state.get(...)` raises AttributeError). Inject
    # a safe `get` implementation on the test session_state so existing
    # code that calls `.get` keeps working in headless tests.
    def _safe_session_get(key, default=None):
        try:
            return at.session_state[key]
        except Exception:
            return default

    try:
        setattr(at.session_state, "get", _safe_session_get)
    except Exception:
        # If we cannot set the attribute for any reason, continue —
        # later reads already use safe indexing fallbacks.
        pass
    # seed dataset and safe defaults (cover common SessionKeys)
    at.session_state[SessionKeys.UPLOADED_DF] = df
    at.session_state[SessionKeys.DF] = df
    at.session_state[SessionKeys.DATASET_NAME] = os.path.basename(path)
    at.session_state[SessionKeys.DATASET_LOADED] = True

    safe_defaults = {
        SessionKeys.AUTONOMOUS_RESULT: None,
        SessionKeys.MODEL_TRAINED: False,
        SessionKeys.RESULTS: {},
        SessionKeys.REPORT_GENERATED: False,
        SessionKeys.SHAP_COMPUTED: False,
        SessionKeys.BEST_MODEL_NAME: None,
        SessionKeys.BEST_MODEL: None,
        SessionKeys.CONFIDENCE_SCORE: None,
        SessionKeys.EDA_GENERATED: False,
        SessionKeys.TARGET_COLUMN: None,
        SessionKeys.X_DATA: None,
        SessionKeys.Y_DATA: None,
        SessionKeys.SHAP_VALUES: None,
        SessionKeys.SHAP_IMPORTANCE: None,
        SessionKeys.AI_INSIGHTS: None,
        SessionKeys.RECOMMENDATIONS: None,
        SessionKeys.PROBLEM_TYPE: "Classification",
    }
    for k, v in safe_defaults.items():
        if k not in at.session_state:
            at.session_state[k] = v

    # navigate to AI Command Center and request pipeline run
    at.session_state[SessionKeys.CURRENT_PAGE] = "ai_command_center"
    at.session_state["pipeline_running"] = True
    at.session_state["pipeline_executed"] = False
    at.session_state["pipeline_progress"] = 0
    at.session_state["pipeline_events"] = []

    start = time.time()
    try:
        at.run(timeout=timeout)
        end = time.time()
        rec["status"] = "COMPLETED"
        rec["runtime"] = end - start
        # Safe reads from AppTest.session_state: use indexing with fallbacks
        try:
            rec["pipeline_progress"] = at.session_state["pipeline_progress"]
        except Exception:
            rec["pipeline_progress"] = None
        try:
            rec["pipeline_events"] = at.session_state["pipeline_events"]
        except Exception:
            rec["pipeline_events"] = []
        try:
            rec["autonomous_result_present"] = bool(at.session_state[SessionKeys.AUTONOMOUS_RESULT])
        except Exception:
            rec["autonomous_result_present"] = False
        try:
            rec["best_model_name"] = at.session_state[SessionKeys.BEST_MODEL_NAME]
        except Exception:
            rec["best_model_name"] = None
        try:
            rec["results"] = at.session_state[SessionKeys.RESULTS]
        except Exception:
            rec["results"] = None
        try:
            rec["report_generated"] = bool(at.session_state[SessionKeys.REPORT_GENERATED])
        except Exception:
            rec["report_generated"] = False
    except Exception as exc:
        end = time.time()
        rec["status"] = "FAILED"
        rec["runtime"] = time.time() - start
        rec["error"] = traceback.format_exc()

    return rec


def main():
    timeout = SMART_MODE_BUDGET_SECONDS + 60
    results = []
    for name, path in DATASETS:
        print(f"Running dataset: {name} ({path}) with timeout={timeout}s")
        r = run_dataset(name, path, timeout)
        results.append(r)
        print(f"  -> {r['status']} (runtime={r.get('runtime')})")

    # write final report
    out_lines = ["# Final E2E Validation Report", ""]
    for r in results:
        out_lines.append(f"## {r['dataset']}")
        out_lines.append(f"- Path: {r['path']}")
        out_lines.append(f"- Status: {r['status']}")
        out_lines.append(f"- Runtime: {r.get('runtime')}")
        out_lines.append(f"- Best model: {r.get('best_model_name')}")
        out_lines.append(f"- Model results: {r.get('results')}")
        out_lines.append(f"- Report generated: {r.get('report_generated')}")
        out_lines.append("- Pipeline events:")
        events = r.get('pipeline_events') or []
        for e in events[-10:]:
            out_lines.append(f"  - {e}")

        if r.get('error'):
            # write full traceback to report and separate log for easier triage
            out_lines.append("- Error:")
            out_lines.append("```")
            out_lines.append(str(r.get('error')))
            out_lines.append("```")
            try:
                log_path = os.path.join(os.getcwd(), f"e2e_error_{r['dataset']}.log")
                with open(log_path, "w", encoding="utf-8") as lf:
                    lf.write(str(r.get('error')))
            except Exception:
                pass
        out_lines.append("")

    report_path = os.path.join(os.getcwd(), "FINAL_E2E_VALIDATION_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out_lines))

    print(f"Wrote final report to {report_path}")


if __name__ == "__main__":
    main()

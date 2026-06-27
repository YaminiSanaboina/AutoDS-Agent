"""Iterate through UI pages and run a headless smoke test for each.

Generates `UI_VALIDATION_REPORT.md` in the repo root summarizing PASS/FAIL
and any exception messages. This is conservative: it ensures pages import
and render without uncaught exceptions under an empty session state.
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import Dict, List

from streamlit.testing.v1.app_test import AppTest

sys.path.insert(0, os.getcwd())
from utils.session_manager import SessionKeys


PAGES = [
    ("AI Command Center", "ai_command_center"),
]


DEFAULT_TIMEOUT = 30.0


def safe_defaults() -> Dict:
    return {
        SessionKeys.CURRENT_PAGE: "ai_command_center",
        SessionKeys.DATASET_LOADED: False,
        SessionKeys.AUTONOMOUS_RESULT: None,
        SessionKeys.MODEL_TRAINED: False,
        SessionKeys.BEST_MODEL: None,
        SessionKeys.BEST_MODEL_NAME: None,
        SessionKeys.RESULTS: {},
        SessionKeys.REPORT_GENERATED: False,
        "pipeline_running": False,
        "pipeline_executed": False,
    }


def run_page_check(page_key: str, timeout: float) -> (str, str):
    at = AppTest.from_file("app.py", default_timeout=timeout)
    # seed safe defaults
    for k, v in safe_defaults().items():
        try:
            at.session_state[k] = v
        except Exception:
            pass

    # set page under test
    at.session_state[SessionKeys.CURRENT_PAGE] = page_key

    try:
        at.run(timeout=timeout)
        return "PASS", ""
    except Exception as exc:
        tb = traceback.format_exc()
        return "FAIL", tb


def main(timeout: float = DEFAULT_TIMEOUT) -> int:
    results: List[Dict] = []
    for display_name, key in PAGES:
        print(f"Testing page: {display_name} ({key})...")
        status, info = run_page_check(key, timeout)
        results.append({"page": display_name, "key": key, "status": status, "error": info})
        print(f"  -> {status}")

    report_lines: List[str] = []
    report_lines.append("# UI Validation Report")
    report_lines.append("")
    report_lines.append("| Page | Key | Status | Error Message | Required Fixes |")
    report_lines.append("|---|---|---|---|---|")
    for r in results:
        error_msg = r["error"].splitlines()[0] if r["error"] else ""
        fixes = "" if r["status"] == "PASS" else "Investigate exception and add safe session_state defaults or guard clauses."
        report_lines.append(f"| {r['page']} | {r['key']} | {r['status']} | {error_msg} | {fixes} |")

    report_path = os.path.join(os.getcwd(), "UI_VALIDATION_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines))

    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

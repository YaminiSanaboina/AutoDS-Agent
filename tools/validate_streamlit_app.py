"""Headless Streamlit validation harness using st.testing.v1.AppTest.

This script attempts to run the top-level `app.py` via AppTest while
pre-seeding a conservative session state to avoid launching long-running
workflows. On timeout or error it prints diagnostics to aid debugging.

Usage: python tools/validate_streamlit_app.py [--timeout SECONDS]
"""
from __future__ import annotations

import argparse
import traceback
from typing import Dict

from streamlit.testing.v1.app_test import AppTest

import os
import sys
# Ensure workspace root is on sys.path so local modules (utils, ui, agents) can be imported
sys.path.insert(0, os.getcwd())

from utils.session_manager import SessionKeys


DEFAULT_TIMEOUT = 120.0


def safe_defaults() -> Dict:
    """Return a conservative mapping of session_state defaults to avoid
    triggering long-running pipelines during headless validation.
    """
    return {
        SessionKeys.CURRENT_PAGE: "ai_command_center",
        SessionKeys.DATASET_LOADED: False,
        SessionKeys.AUTONOMOUS_RESULT: None,
        SessionKeys.MODEL_TRAINED: False,
        SessionKeys.REPORT_GENERATED: False,
        SessionKeys.SHAP_COMPUTED: False,
        SessionKeys.RESULTS: {},
        SessionKeys.EDA_GENERATED: False,
        "pipeline_running": False,
        "pipeline_executed": False,
        "pipeline_progress": 0,
    }


def main(timeout: float | None = None) -> int:
    timeout = timeout or DEFAULT_TIMEOUT

    print(f"Creating AppTest.from_file('app.py') (default_timeout={timeout})")
    at = AppTest.from_file("app.py", default_timeout=timeout)

    # Seed safe session state values
    print("Seeding safe session_state defaults...")
    defaults = safe_defaults()
    for k, v in defaults.items():
        try:
            at.session_state[k] = v
        except Exception:
            # Best-effort; continue even if some keys are not assignable
            print(f"Warning: failed to set session key {k}")

    print("Running headless app_test.run()... (this may take a few seconds)")
    try:
        at.run(timeout=timeout)
        print("AppTest run completed successfully.")
        # Basic diagnostic summary
        main_elems = at.main
        sidebar_elems = at.sidebar
        print(f"Main elements count: {len(list(main_elems._elements)) if hasattr(main_elems, '_elements') else 'unknown'}")
        print(f"Sidebar elements count: {len(list(sidebar_elems._elements)) if hasattr(sidebar_elems, '_elements') else 'unknown'}")
        return 0
    except Exception as exc:  # pragma: no cover - diagnostics path
        print("AppTest failed:")
        traceback.print_exc()
        # Surface any helpful runner attributes when available
        try:
            print("Attempting to print query_params and secrets (if present):")
            print("query_params:", getattr(at, "query_params", None))
            print("secrets:", getattr(at, "secrets", None))
        except Exception:
            pass
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()
    raise SystemExit(main(timeout=args.timeout))

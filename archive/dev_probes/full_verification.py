import os
import sys
import json
import time
import traceback
from pathlib import Path
from streamlit.testing.v1.app_test import AppTest
import pandas as pd

# Ensure repo root importability
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from agents.master_autonomous_pipeline import MasterAutonomousPipeline
from utils.session_manager import SessionKeys

DATASETS = [
    ("Iris", Path("data") / "Iris.csv"),
    ("Wine-Quality", Path("data") / "Wine-Quality.csv"),
    ("Housing", Path("data") / "Housing.csv"),
]

OUT_DIR = Path("verification_outputs")
OUT_DIR.mkdir(exist_ok=True)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

summary = []

for name, path in DATASETS:
    rec = {"dataset": name, "path": str(path), "ui_checks": {}, "pipeline": {}, "errors": []}
    print(f"\n=== Verifying {name} ({path}) ===")

    # Part A: AppTest upload + session persistence
    try:
        at = AppTest.from_file("app.py", default_timeout=120)
    except Exception as exc:
        rec["errors"].append(f"AppTest.from_file failed: {exc}")
        summary.append(rec)
        continue

    try:
        # provide safe .get api on AppTest.session_state for headless reads
        def _safe_session_get(key, default=None):
            try:
                return at.session_state[key]
            except Exception:
                return default
        try:
            setattr(at.session_state, 'get', _safe_session_get)
        except Exception:
            pass

        # initial run to render widgets
        at.run(timeout=60)
    except Exception as exc:
        rec["errors"].append(f"Initial at.run() failed: {exc}")

    try:
        if not hasattr(at, 'file_uploader') or len(at.file_uploader) == 0:
            rec["ui_checks"]["uploader_found"] = False
            print("No file uploader widgets found in AppTest render.")
        else:
            rec["ui_checks"]["uploader_found"] = True
            uploader = at.file_uploader[0]
            content = path.read_bytes()
            uploader.set_value((path.name, content, 'text/csv'))
            # run to process upload
            at.run(timeout=60)

            # inspect internal safe session state
            state = getattr(at.session_state, '_state', {})
            keys = list(at.session_state._keys()) if hasattr(at.session_state, '_keys') else []
            rec["ui_checks"]["session_keys"] = keys
            rec["ui_checks"]["dataset_loaded_after_upload"] = bool(state.get('dataset_loaded'))
            rec["ui_checks"]["uploaded_dataframe_exists_after_upload"] = state.get('uploaded_dataframe') is not None
            rec["ui_checks"]["df_exists_after_upload"] = state.get('df') is not None

            # simulate rerun
            at.run(timeout=10)
            state2 = getattr(at.session_state, '_state', {})
            rec["ui_checks"]["dataset_loaded_after_rerun"] = bool(state2.get('dataset_loaded'))
            rec["ui_checks"]["uploaded_dataframe_exists_after_rerun"] = state2.get('uploaded_dataframe') is not None
            rec["ui_checks"]["df_exists_after_rerun"] = state2.get('df') is not None

            # try to detect run button and its disabled flag
            btn_disabled = None
            try:
                # Inspect at.main._elements for Button widgets
                btns = []
                main = getattr(at, 'main', None)
                if main is not None and hasattr(main, '_elements'):
                    for idx, el in list(main._elements.items()):
                        try:
                            if getattr(el, 'type', '').lower() == 'button' or getattr(el, 'label', '') == 'Run Autonomous Analysis':
                                btns.append({'idx': idx, 'label': getattr(el, 'label', None), 'disabled': getattr(el, 'disabled', None)})
                        except Exception:
                            continue
                rec["ui_checks"]["found_run_buttons"] = btns
            except Exception as exc:
                rec["ui_checks"]["found_run_buttons_error"] = str(exc)

    except Exception as exc:
        tb = traceback.format_exc()
        rec["errors"].append(f"Upload processing error: {exc}\n{tb}")

    # Save UI snapshot
    ui_out = OUT_DIR / f"{name}_ui_snapshot.json"
    ui_out.write_text(json.dumps(rec["ui_checks"], default=str, indent=2))
    print("UI checks written to", ui_out)

    # Part B: Run pipeline synchronously via MasterAutonomousPipeline
    try:
        print("Running synchronous pipeline (may take a while)...")
        df = pd.read_csv(path)
        pipeline = MasterAutonomousPipeline()

        events = []

        def progress_cb(ev):
            t = time.time()
            print(f"PROGRESS: {ev}")
            events.append({'time': t, 'ev': ev})

        out = pipeline.run_pipeline(dataset=df, dataset_name=path.name, project_goal=f"Validate {name}", progress_callback=progress_cb, smart_mode=True, max_seconds=600, use_cache=False)
        # write output
        out_path = OUT_DIR / f"{name}_pipeline_output.json"
        out_path.write_text(json.dumps(out, default=str, indent=2))
        rec["pipeline"]["output_path"] = str(out_path)
        rec["pipeline"]["events"] = events
        # capture expected keys
        rec["pipeline"]["best_model_name"] = out.get('best_model') or out.get('best_model_name') or out.get('model')
        rec["pipeline"]["training_artifacts_present"] = bool(out.get('training_artifacts') or out.get('model_results') or out.get('results'))
        rec["pipeline"]["trust_score"] = out.get('trust_score') or out.get('deployment_readiness_score') or out.get('trust_score')

        # Generate PDF report if pipeline returned a structured payload or final report data
        try:
            from agents.report_agent import generate_pdf_report
            payload = out.get('report_payload') or out.get('final_report') or out
            pdf_path = generate_pdf_report(payload)
            rec["pipeline"]["pdf_generated"] = pdf_path
        except Exception as exc:
            rec["pipeline"]["pdf_error"] = str(exc)

    except Exception as exc:
        tb = traceback.format_exc()
        rec["errors"].append(f"Pipeline execution error: {exc}\n{tb}")

    summary.append(rec)

# write final summary
summary_path = OUT_DIR / "verification_summary.json"
summary_path.write_text(json.dumps(summary, default=str, indent=2))
print('\nVerification summary written to', summary_path)
print(json.dumps(summary, default=str, indent=2))

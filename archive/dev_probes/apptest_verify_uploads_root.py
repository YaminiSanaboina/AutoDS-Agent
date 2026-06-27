import os
import sys
from pathlib import Path
from streamlit.testing.v1.app_test import AppTest

# Ensure repo root is importable inside AppTest script execution
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

for name in ["Iris.csv", "Wine-Quality.csv", "Housing.csv"]:
    print(f"=== {name} ===")
    path = ROOT / "data" / name
    if not path.exists():
        print("MISSING", path)
        continue

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    # AppTest SafeSessionState does not expose .get by default
    try:
        setattr(at.session_state, "get", lambda key, default=None: at.session_state._state.get(key, default))
    except Exception:
        pass

    try:
        at.run(timeout=30)
    except Exception as exc:
        print("Initial run failed", exc)
        continue

    if not hasattr(at, 'file_uploader') or len(at.file_uploader) == 0:
        print('No file uploader widgets found')
        continue

    uploader = at.file_uploader[0]
    try:
        uploader.set_value((name, path.read_bytes(), 'text/csv'))
    except Exception as exc:
        print('set_value failed', exc)
        continue

    try:
        at.run(timeout=30)
    except Exception as exc:
        print('Second run failed', exc)
        # still inspect session_state if possible

    state = getattr(at.session_state, '_state', None)
    if state is None:
        print('No internal session state available')
        continue

    print('keys', list(at.session_state._keys()))
    print('dataset_loaded', state.get('dataset_loaded'))
    print('uploaded_dataframe_exists', state.get('uploaded_dataframe') is not None)
    print('df_exists', state.get('df') is not None)
    df = state.get('df')
    print('df_shape', df.shape if df is not None else None)
    print()

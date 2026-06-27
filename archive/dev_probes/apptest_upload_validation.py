import os
import sys
from pathlib import Path
from streamlit.testing.v1.app_test import AppTest

sys.path.insert(0, os.getcwd())

# safe custom get for AppTest SafeSessionState

def safe_get(key, default=None):
    try:
        return at.session_state[key]
    except Exception:
        return default

for name in ["Iris.csv", "Wine-Quality.csv", "Housing.csv"]:
    print(f"=== {name} ===")
    path = Path("data") / name
    if not path.exists():
        print("MISSING", path)
        continue
    content = path.read_bytes()
    at = AppTest.from_file("app.py", default_timeout=30)
    try:
        setattr(at.session_state, "get", lambda key, default=None: at.session_state._state.get(key, default))
    except Exception as exc:
        print("Warning: could not attach get():", exc)
    at.run(timeout=30)
    if not hasattr(at, 'file_uploader') or len(at.file_uploader) == 0:
        print('No file uploader widgets found')
        continue
    uploader = at.file_uploader[0]
    uploader.set_value((name, content, 'text/csv'))
    at.run(timeout=30)
    try:
        print('dataset_loaded', at.session_state._state.get('dataset_loaded'))
    except Exception as exc:
        print('dataset_loaded error', exc)
    try:
        print('uploaded_dataframe_exists', at.session_state._state.get('uploaded_dataframe') is not None)
    except Exception as exc:
        print('uploaded_dataframe_exists error', exc)
    try:
        print('df_exists', at.session_state._state.get('df') is not None)
    except Exception as exc:
        print('df_exists error', exc)
    try:
        df = at.session_state._state.get('df')
        print('df_shape', df.shape if df is not None else None)
    except Exception as exc:
        print('df_shape error', exc)
    print()

from pathlib import Path
from streamlit.testing.v1.app_test import AppTest

datasets = ["Iris.csv", "Wine-Quality.csv", "Housing.csv"]
for name in datasets:
    print(f"=== {name} ===")
    p = Path("data") / name
    if not p.exists():
        print("MISSING", p)
        continue
    content = p.read_bytes()
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run(timeout=30)
    if not hasattr(at, 'file_uploader') or len(at.file_uploader) == 0:
        print("No file uploader widgets found")
        continue
    uploader = at.file_uploader[0]
    uploader.set_value((name, content, 'text/csv'))
    at.run(timeout=30)
    try:
        print('dataset_loaded', at.session_state['dataset_loaded'])
    except Exception as exc:
        print('dataset_loaded error', repr(exc))
    try:
        print('uploaded_dataframe_exists', 'uploaded_dataframe' in at.session_state and at.session_state['uploaded_dataframe'] is not None)
    except Exception as exc:
        print('uploaded_dataframe_exists error', repr(exc))
    try:
        print('df_exists', 'df' in at.session_state and at.session_state['df'] is not None)
    except Exception as exc:
        print('df_exists error', repr(exc))
    try:
        df_shape = at.session_state['df'].shape if 'df' in at.session_state else None
        print('df_shape', df_shape)
    except Exception as exc:
        print('df_shape error', repr(exc))
    print()

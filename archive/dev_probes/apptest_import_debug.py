from pathlib import Path
from streamlit.testing.v1.app_test import AppTest
from pathlib import Path

script_path = Path('tmp_import_test.py')
script_path.write_text('import os, sys\nprint("pwd", os.getcwd())\nprint("sys.path", sys.path)\nimport config\nprint("config ok", config.APP_TITLE)\n', encoding='utf-8')
print('script written', script_path.absolute())

at = AppTest.from_file(str(script_path), default_timeout=5)
print('AppTest created', at)
try:
    result = at.run(timeout=5)
    print('AppTest run returned', result)
except Exception as exc:
    import traceback
    traceback.print_exc()

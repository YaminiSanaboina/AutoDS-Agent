import py_compile
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
files = [ROOT / "app.py"] + list((ROOT / "ui").rglob("*.py")) + list((ROOT / "utils").rglob("*.py")) + list((ROOT / "agents").rglob("*.py"))
failed = []
for file_path in files:
    try:
        py_compile.compile(str(file_path), doraise=True)
    except Exception as exc:
        print(f"FAILED: {file_path}: {exc}")
        failed.append((str(file_path), str(exc)))
if failed:
    sys.exit(1)
print(f"All {len(files)} files compiled successfully")

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
patterns = [
    r"file_uploader\(",
    r"SessionKeys\.DATASET_LOADED",
    r"SessionKeys\.DF",
    r"SessionKeys\.UPLOADED_DF",
    r"st\.session_state\[\"df\"\]",
    r"has_dataset\(",
    r"get_dataframe\(",
    r"Dataset Status",
    r"Health Score",
    r"dataset_status",
    r"st\.session_state\[.*SessionKeys\.DATASET_LOADED.*\]",
    r"st\.session_state\[.*SessionKeys\.DF.*\]",
    r"st\.session_state\[.*SessionKeys\.UPLOADED_DF.*\]",
]

def find_matches():
    rows = []
    for path in ROOT.rglob('*.py'):
        try:
            text = path.read_text(encoding='utf-8')
        except Exception:
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            for pat in patterns:
                if re.search(pat, line):
                    rows.append((str(path.relative_to(ROOT)), idx, line.strip()))
                    break
    return rows

if __name__ == '__main__':
    rows = find_matches()
    rows.sort()
    for file, line, code in rows:
        print(f"{file}:{line}: {code}")

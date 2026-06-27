import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
patterns = [
    ('SessionKeys.DATASET_LOADED', re.compile(r'SessionKeys\.DATASET_LOADED')),
    ('SessionKeys.UPLOADED_DF', re.compile(r'SessionKeys\.UPLOADED_DF')),
    ('SessionKeys.DF', re.compile(r'SessionKeys\.DF')),
    ("st.session_state['df']", re.compile(r"st\.session_state\[['\"]df['\"]\]")),
    ("st.session_state.get('df')", re.compile(r"st\.session_state\.get\(['\"]df['\"]\)")),
    ('get_dataframe()', re.compile(r'get_dataframe\(')),
    ('has_dataset()', re.compile(r'has_dataset\(')),
]

rows = []
for path in sorted(ROOT.rglob('*.py')):
    if 'site-packages' in str(path):
        continue
    try:
        text = path.read_text(encoding='utf-8')
    except Exception:
        continue
    for idx, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        for name, pat in patterns:
            if pat.search(line):
                rows.append((name, str(path.relative_to(ROOT)), idx, stripped))
                break

for name, _ in patterns:
    print(f'=== {name} ===')
    for entry in rows:
        if entry[0] == name:
            print(f'{entry[1]}:{entry[2]}: {entry[3]}')

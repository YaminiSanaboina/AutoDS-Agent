import re
import os
from collections import defaultdict

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
key_re = re.compile(r"key\s*=\s*['\"]([^'\"]+)['\"]")
sstate_re = re.compile(r"st\.session_state\[(?:'|\")([^'\"]+)(?:'|\")\]")

keys = defaultdict(list)
sstates = defaultdict(list)

for dirpath, dirnames, filenames in os.walk(root):
    # skip virtual envs, .git, .pytest_cache
    if any(p in dirpath for p in ['.git', 'site-packages', '__pycache__', '.pytest_cache', 'venv']):
        continue
    for fn in filenames:
        if not fn.endswith('.py'):
            continue
        fp = os.path.join(dirpath, fn)
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                txt = f.read()
        except Exception:
            continue
        for m in key_re.finditer(txt):
            keys[m.group(1)].append(fp)
        for m in sstate_re.finditer(txt):
            sstates[m.group(1)].append(fp)

print('Found widget keys:')
for k, locs in sorted(keys.items(), key=lambda x: (-len(x[1]), x[0])):
    if len(locs) > 1:
        print(f"DUPLICATE KEY ({len(locs)}): {k}")
        for l in locs:
            print(f"  - {os.path.relpath(l, root)}")

print('\nFound session_state keys used in st.session_state[...]')
for k, locs in sorted(sstates.items(), key=lambda x: (-len(x[1]), x[0])):
    if len(locs) > 1:
        print(f"SESSION KEY USED IN MULTIPLE FILES ({len(locs)}): {k}")
        for l in locs:
            print(f"  - {os.path.relpath(l, root)}")

print('\nSummary counts:')
print('Total distinct widget keys:', len(keys))
print('Total distinct session_state keys:', len(sstates))

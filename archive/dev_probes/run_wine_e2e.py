import os, sys
sys.path.insert(0, os.getcwd())
from tools.e2e_validate import run_dataset

result = run_dataset('Wine', 'data/Wine-Quality.csv', timeout=180)
print(result)

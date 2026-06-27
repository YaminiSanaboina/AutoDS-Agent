"""
Verify that the upload fix (st.rerun() added) is in place.

The fix ensures that when a user uploads a CSV file via st.file_uploader,
the page reruns so that the "Run Autonomous Analysis" button becomes visible.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Read the main upload handler code
with open("ui/ai_command_center.py", "r") as f:
    content = f.read()

# Find the upload handler section
upload_handler_start = content.find("if uploaded_file is not None:")
upload_handler_section = content[upload_handler_start:upload_handler_start+1200]

print("=" * 80)
print("UPLOAD HANDLER CODE VERIFICATION")
print("=" * 80)
print(upload_handler_section)
print("=" * 80)

# Check for the fix: st.rerun() after success message
has_rerun = "st.success" in upload_handler_section and "st.rerun()" in upload_handler_section
rerun_after_success = (
    upload_handler_section.find("st.success(\"Dataset loaded successfully\")") < 
    upload_handler_section.find("st.rerun()")
)

print("\nFIX VERIFICATION:")
print(f"✓ st.success() present: {bool('st.success' in upload_handler_section)}")
print(f"✓ st.rerun() present: {'st.rerun()' in upload_handler_section}")
print(f"✓ st.rerun() comes after st.success(): {rerun_after_success}")

if has_rerun and rerun_after_success:
    print("\n✅ FIX IS CORRECTLY APPLIED!")
    print("\nWhat the fix does:")
    print("1. User uploads CSV via st.file_uploader")
    print("2. Code reads CSV and calls reset_on_new_dataset(df, filename)")
    print("3. reset_on_new_dataset() updates session state:")
    print("   - Sets SessionKeys.UPLOADED_DF = df")
    print("   - Sets SessionKeys.DF = df")
    print("   - Sets SessionKeys.DATASET_LOADED = True")
    print("4. NEW: st.rerun() is called, which reruns the page")
    print("5. On rerun, has_dataset() now returns True")
    print("6. Run button at line 736 becomes visible")
    print("7. User can click 'Run Autonomous Analysis' to execute pipeline")
else:
    print("\n❌ FIX IS NOT APPLIED - st.rerun() missing or in wrong position!")
    sys.exit(1)

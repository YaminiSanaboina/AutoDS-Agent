import py_compile
import glob
import importlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LEGACY_PAGES = [
    "ui.legacy_pages.dashboard",
    "ui.legacy_pages.data_hub",
    "ui.legacy_pages.dataset_library",
    "ui.legacy_pages.cleaning_lab",
    "ui.legacy_pages.eda_explorer",
    "ui.legacy_pages.automl",
    "ui.legacy_pages.decision_intelligence",
    "ui.legacy_pages.prediction_playground",
    "ui.legacy_pages.report_center",
    "ui.legacy_pages.deployment_center",
    "ui.legacy_pages.model_registry",
    "ui.legacy_pages.ethics_governance",
    "ui.legacy_pages.documentation_hub",
    "ui.legacy_pages.ai_assistant",
    "ui.legacy_pages.agent_activity_monitor",
]

MODULAR_PAGES = [
    "ui.pages.home",
    "ui.pages.upload",
    "ui.pages.workspace",
    "ui.pages.cleaning",
    "ui.pages.eda",
    "ui.pages.automl",
    "ui.pages.xai",
    "ui.pages.report",
]

TOP_LEVEL_UI = [
    "ui.ai_command_center",
    "ui.report_center",
    "ui.system_health_dashboard",
]


@pytest.mark.parametrize("module_name", LEGACY_PAGES + MODULAR_PAGES + TOP_LEVEL_UI)
def test_ui_module_imports(module_name):
    module = importlib.import_module(module_name)
    assert hasattr(module, "render"), f"{module_name} must expose render()"


def test_all_ui_files_compile():
    files = glob.glob(str(PROJECT_ROOT / "ui" / "**" / "*.py"), recursive=True)
    failed = []
    for file_path in files:
        try:
            py_compile.compile(file_path, doraise=True)
        except Exception as exc:
            failed.append((file_path, str(exc)))
    assert not failed, f"UI compile failures: {failed}"

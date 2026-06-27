"""Shared pytest fixtures and helpers for AutoDS-Agent tests."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AI_ASSISTANT_PATH = PROJECT_ROOT / "ui" / "legacy_pages" / "ai_assistant.py"


def load_ai_assistant_module():
    """Load the legacy AI assistant module without importing the full ui package."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    spec = importlib.util.spec_from_file_location("ai_assistant", AI_ASSISTANT_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"AI assistant module not found: {AI_ASSISTANT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ai_assistant():
    return load_ai_assistant_module()


@pytest.fixture
def session_ready():
    """Initialize Streamlit session state for assistant integration tests."""
    import config

    config.LLM_PROVIDER = "fallback"
    from utils.session_manager import init_session

    init_session()
    if "llm_agent" in st.session_state:
        del st.session_state["llm_agent"]
    return True


def set_test_dataset(df: pd.DataFrame, name: str) -> None:
    from utils.session_manager import set_dataframe

    set_dataframe(df, name)

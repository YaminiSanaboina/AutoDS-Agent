"""AI Report Center — delegates to the Interactive Report Center experience."""

from __future__ import annotations

import streamlit as st

from ui.components import require_dataset
from ui.interactive_report_center import render_interactive_report_center
from utils.session_manager import get_autonomous_result, has_autonomous_result
from utils.styles import render_hero


def render():
    """Render the premium Interactive AI Report Center."""
    render_hero("Interactive AI Report Center", "Browser-first executive reports with optional PDF export")

    if not require_dataset():
        return

    output = get_autonomous_result() if has_autonomous_result() else None
    render_interactive_report_center(output)

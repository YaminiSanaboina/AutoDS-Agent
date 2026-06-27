"""Global error boundary helpers for Streamlit UI panels."""

from __future__ import annotations

import traceback
from typing import Any, Callable, Optional

import streamlit as st


def log_exception(panel_name: str, exc: Exception) -> None:
    """Log panel exceptions for debugging without crashing the app."""
    print(f"[AutoDS UI][{panel_name}] {type(exc).__name__}: {exc}")
    print(traceback.format_exc())


def render_panel_error(panel_name: str, exc: Exception) -> None:
    """Render a friendly Streamlit warning when a panel fails."""
    st.warning(
        f"**{panel_name}** could not be rendered completely. "
        f"({type(exc).__name__}: {exc})"
    )
    st.caption("Other sections of the Command Center remain available.")


def run_panel(panel_name: str, render_fn: Callable[[], Any]) -> None:
    """Execute a panel render function inside an error boundary."""
    try:
        render_fn()
    except Exception as exc:
        log_exception(panel_name, exc)
        render_panel_error(panel_name, exc)

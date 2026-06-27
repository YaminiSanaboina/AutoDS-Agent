import streamlit as st

from config import APP_ICON, APP_TITLE
from utils.session_manager import init_session
import utils.styles as styles

from ui.ai_command_center import render as render_ai_command_center

print("[STARTUP] app.py imported")


def main():
    print("[STARTUP] app.py main started")
    # Streamlit page configuration
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Initialize global styles and session
    styles.inject_styles()
    init_session()
    print("[STARTUP] session initialized")

    # No developer preloads: UI starts with a clean session.

    # Render the single AI Command Center page
    render_ai_command_center()
    print("[STARTUP] ai_command_center render called")


if __name__ == "__main__":
    main()

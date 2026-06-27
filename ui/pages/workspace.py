import streamlit as st

from agents.dataset_agent import analyze_dataset
from ui.components import dataset_loaded_banner, health_badge, require_dataset
from utils.health_score import compute_health_score
from utils.session_manager import get_dataframe, get_dataset_name
from utils.styling import render_page_header


def render():
    render_page_header(
        "Dataset Workspace",
        "Comprehensive profiling with AI health scoring",
    )

    if not require_dataset():
        return

    df = get_dataframe()
    dataset_loaded_banner(df, get_dataset_name())

    try:
        health = compute_health_score(df)
        analysis = analyze_dataset(df)
    except Exception as exc:
        st.warning(f"Workspace analysis unavailable: {exc}")
        return

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        st.metric("Rows", f"{analysis['rows']:,}")
    with c2:
        st.metric("Columns", analysis["columns"])
    with c3:
        st.metric("Duplicates", analysis["duplicate_rows"])
    with c4:
        st.markdown("##### Dataset Health")
        health_badge(f"{health['score']}/100 — {health['grade']}", health["grade_class"])
        st.caption(health["summary"])

    st.markdown("---")
    st.subheader("Health Breakdown")
    breakdown_cols = st.columns(4)
    for idx, (label, value) in enumerate(health["breakdown"].items()):
        with breakdown_cols[idx]:
            st.metric(label, f"{value}/100")

    tab_overview, tab_types, tab_missing, tab_stats = st.tabs(
        ["Overview", "Data Types", "Missing Values", "Statistics"]
    )

    with tab_overview:
        st.write("**Column Names:**")
        st.code(", ".join(analysis["column_names"]), language=None)
        st.dataframe(df.head(15), width="stretch", hide_index=True)

    with tab_types:
        types_df = analysis["data_types"].reset_index()
        types_df.columns = ["Column", "Data Type"]
        st.dataframe(types_df, width="stretch", hide_index=True)

    with tab_missing:
        missing_df = analysis["missing_values"].reset_index()
        missing_df.columns = ["Column", "Missing Count"]
        missing_df["Missing %"] = (
            missing_df["Missing Count"] / len(df) * 100
        ).round(2)
        st.dataframe(missing_df, width="stretch", hide_index=True)

    with tab_stats:
        st.dataframe(analysis["summary"], width="stretch")

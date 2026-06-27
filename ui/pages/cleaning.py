import streamlit as st

from agents.cleaning_agent import clean_dataset
from ui.components import dataset_loaded_banner, issue_card, require_dataset
from utils.health_score import compute_health_score, detect_data_issues
from utils.session_manager import SessionKeys, get_dataframe, get_dataset_name, set_dataframe
from utils.styling import render_page_header


def render():
    render_page_header(
        "AI Data Cleaning Center",
        "Automated issue detection with intelligent recommendations",
    )

    if not require_dataset():
        return

    df = get_dataframe()
    dataset_loaded_banner(df, get_dataset_name())

    if st.session_state.get(SessionKeys.CLEANING_ISSUES) is None:
        st.session_state[SessionKeys.CLEANING_ISSUES] = detect_data_issues(df)

    issues = st.session_state.get(SessionKeys.CLEANING_ISSUES) or []
    health_before = compute_health_score(df)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Issues Detected", len(issues))
    with c2:
        st.metric("Pre-Clean Health", f"{health_before['score']}/100")

    st.subheader("Detected Issues & Recommendations")
    for issue in issues:
        issue_card(
            issue["title"],
            issue["description"],
            issue["recommendation"],
            issue["severity"],
        )

    st.markdown("---")

    if st.button("Run AI Cleaning Pipeline", type="primary", width="content"):
        try:
            cleaned_df, report = clean_dataset(df)
            set_dataframe(cleaned_df, get_dataset_name())
            st.session_state[SessionKeys.CLEANING_REPORT] = report
            st.session_state[SessionKeys.CLEANING_ISSUES] = detect_data_issues(cleaned_df)
            st.session_state[SessionKeys.CLEANING_ISSUES] = [
                item for item in st.session_state[SessionKeys.CLEANING_ISSUES]
                if item["title"] != "No Critical Issues"
            ] or st.session_state[SessionKeys.CLEANING_ISSUES]
            st.rerun()
        except Exception as exc:
            st.error(f"Cleaning failed: {exc}")

    if st.session_state.get(SessionKeys.CLEANING_REPORT):
        health_after = compute_health_score(get_dataframe())
        st.success("Cleaning completed successfully!")
        st.metric(
            "Post-Clean Health",
            f"{health_after['score']}/100",
            delta=f"{health_after['score'] - health_before['score']:.1f}",
        )

        st.subheader("Cleaning Actions")
        for item in st.session_state[SessionKeys.CLEANING_REPORT]:
            st.write(f"✅ {item}")

        st.subheader("Cleaned Data Preview")
        st.dataframe(
            get_dataframe().head(10),
            width="stretch",
            hide_index=True,
        )

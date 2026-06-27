import streamlit as st

from ui.components import glass_metric, glass_panel, glass_panel_small
from utils.session_manager import (
    SessionKeys,
    get_dataframe,
    has_dataset,
    get_dataset_name,
)
from utils.health_score import compute_health_score
from utils.styles import render_hero

# AI Workflow agent
from agents.workflow_agent import AIWorkflowAgent
 


def render():
    render_hero(
        "AutoDS Agent",
        "AI-Powered Autonomous Data Scientist",
    )
    # Initialize AI workflow agent
    workflow_agent = AIWorkflowAgent()

    # Intelligent Welcome Assistant Card
    wf_summary = workflow_agent.analyze_workflow()
    progress_info = workflow_agent.calculate_progress()
    percent = progress_info.get("percentage", 0)
    label = progress_info.get("label", "Beginner")

    # Build an intelligent greeting based on workflow state
    if not has_dataset():
        greeting_title = "Welcome to AutoDS Agent"
        greeting_body = (
            "Upload your first dataset and I will guide you through the complete AI workflow."
        )
    else:
        ds_name = get_dataset_name()
        greeting_title = f"Welcome back"
        greeting_body = f"Your dataset '{ds_name}' is ready. You have completed {percent}% of your AI project ({label})."

    glass_panel(greeting_title, greeting_body)

    df = get_dataframe()
    health = compute_health_score(df) if has_dataset() else None

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        glass_metric("", "Dataset Status", "Loaded" if has_dataset() else "None")
    with c2:
        glass_metric("", "Features", str(df.shape[1]) if has_dataset() else "—")
    with c3:
        glass_metric(
            "", "Quality Score",
            f"{health['score']}/100" if health else "—",
        )
    with c4:
        glass_metric(
            "", "Models Trained",
            str(len(st.session_state.get(SessionKeys.RESULTS) or {}))
            if st.session_state.get(SessionKeys.MODEL_TRAINED) else "0",
        )
    with c5:
        conf = st.session_state.get(SessionKeys.CONFIDENCE_SCORE)
        glass_metric("", "AI Confidence", f"{conf}%" if conf else "—")
    with c6:
        glass_metric(
            "", "Reports",
            "1" if st.session_state.get(SessionKeys.REPORT_GENERATED) else "0",
        )

    # AI Project Status Overview (compact)
    st.markdown("---")
    st.subheader("AI Project Status")
    summary = workflow_agent.get_workflow_summary()
    ds = summary.get("dataset") or {}
    model = summary.get("model") or {}

    status_cols = st.columns([2, 1])
    with status_cols[0]:
        st.write(f"**Project Stage:** {wf_summary.get('current_stage')}")
        st.progress(int(percent))
        st.write(f"**Progress:** {percent}% — {label}")
    with status_cols[1]:
        st.write(f"**Dataset:** {ds.get('name') or '—'}")
        st.write(f"**Rows:** {ds.get('rows') or '—'}")
        st.write(f"**Columns:** {ds.get('columns') or '—'}")
        st.write(f"**Quality:** {ds.get('quality_score') or '—'}")
        st.write(f"**Best model:** {model.get('name') or '—'}")
        st.write(f"**Model score:** {model.get('score') or '—'}")

    st.markdown("---")
    st.subheader("AI Workflow Pipeline")

    steps = [
        ("Upload Data", has_dataset()),
        ("Clean Data", bool(st.session_state.get(SessionKeys.CLEANING_REPORT))),
        ("Analyze Data", st.session_state.get(SessionKeys.EDA_GENERATED, False)),
        ("Train Models", st.session_state.get(SessionKeys.MODEL_TRAINED, False)),
        ("Explain Decisions", st.session_state.get(SessionKeys.SHAP_COMPUTED, False)),
        ("Generate AI Report", st.session_state.get(SessionKeys.REPORT_GENERATED, False)),
    ]

    step_html = ""
    for idx, (label, done) in enumerate(steps):
        css = "done" if done else "active" if (idx == 0 or steps[idx - 1][1]) and not done else ""
        arrow = '<span class="workflow-arrow">→</span>' if idx < len(steps) - 1 else ""
        step_html += f'<div class="workflow-step {css}">{label}</div>{arrow}'

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;">{step_html}</div>',
        unsafe_allow_html=True,
    )

    # AI Next Step Advisor
    st.markdown("---")
    st.subheader("AI Recommendation")
    recommendation = workflow_agent.recommend_next_action()
    glass_panel_small(recommendation)

    # Detailed workflow cards (keeps existing pipeline logic intact)
    st.markdown("---")
    st.markdown("<div style='display:flex;gap:12px;flex-wrap:wrap;'>", unsafe_allow_html=True)
    steps_info = [
        ("Data Hub", "Upload and understand your dataset."),
        ("Data Quality Lab", "Detect and fix missing values, duplicates, and quality issues."),
        ("EDA Explorer", "Discover patterns, correlations, and trends."),
        ("AutoML Studio", "Automatically train and compare machine learning models."),
        ("Decision Intelligence", "Understand why models make predictions using Explainable AI."),
        ("AI Research Report", "Generate a professional business and technical report."),
    ]
    for title, desc in steps_info:
        st.markdown(
            f"<div class='card-card' style='width:260px;padding:0.9rem;'><strong>{title}</strong>"
            f"<div style='margin-top:6px;color:#64748B;font-size:0.95rem;'>{desc}</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Dataset status explanation
    st.markdown("---")
    st.subheader("Dataset Status")
    if not has_dataset():
        st.info(
            "No dataset loaded. Start by uploading a CSV or Excel file in the Data Hub.\n"
            "The Data Hub helps you preview your data, run initial AI analysis, and prepare the dataset for cleaning and modeling."
        )
    else:
        ds_name = get_dataset_name()
        rows = df.shape[0]
        cols = df.shape[1]
        missing_total = int(df.isnull().sum().sum())
        dup_rows = int(df.duplicated().sum())

        st.write(f"**Dataset:** {ds_name}")
        st.write(f"**Rows:** {rows:,} · **Columns:** {cols}")
        st.write(
            f"This dataset contains {rows:,} records and {cols} features. You can clean it in Data Quality Lab, "
            "explore patterns in EDA Explorer, then train models in AutoML Studio."
        )
        if missing_total > 0:
            st.warning(f"Missing values detected: {missing_total} cells. Consider using Data Quality Lab to address them.")
        if dup_rows > 0:
            st.warning(f"Duplicate rows detected: {dup_rows}. Consider removing duplicates in the Data Quality Lab.")

    # Model status and guidance
    st.markdown("---")
    st.subheader("Model Status")
    if not st.session_state.get(SessionKeys.MODEL_TRAINED):
        st.info(
            "No models have been trained yet. Use AutoML Studio to select a target, run automatic training, and compare models.\n"
            "AutoML Studio trains multiple algorithms, evaluates them on held-out data, and suggests the best performing model."
        )
    else:
        best = st.session_state.get(SessionKeys.BEST_MODEL_NAME)
        results = st.session_state.get(SessionKeys.RESULTS) or {}
        if best and best in results:
            score = results[best]
            pt = st.session_state.get(SessionKeys.PROBLEM_TYPE)
            fmt = f"{score:.2%}" if pt == "Classification" else f"{score:.4f}"
            st.write(f"**Best Model:** {best}")
            st.write(f"**Score:** {fmt}")
            if pt == "Classification":
                st.write(f"An accuracy of {fmt} means the model correctly predicts approximately {int(score*100)} out of 100 cases.")
            else:
                st.write(f"An R² of {fmt} means the model explains about {score*100:.1f}% of the variance in the target.")
        else:
            st.info("Modeling results are not yet available. Run AutoML Studio to train and evaluate models.")

    # Smart Action Shortcuts
    st.markdown("---")
    st.subheader("Quick Actions")
    col_a, col_b, col_c, col_d = st.columns(4)
    if not has_dataset():
        if col_a.button("Go to Data Hub"):
            st.session_state[SessionKeys.CURRENT_PAGE] = "data_hub"
    elif not st.session_state.get(SessionKeys.MODEL_TRAINED):
        if col_b.button("Open AutoML Studio"):
            st.session_state[SessionKeys.CURRENT_PAGE] = "automl"
    elif st.session_state.get(SessionKeys.MODEL_TRAINED) and not (st.session_state.get(SessionKeys.SHAP_COMPUTED) or st.session_state.get(SessionKeys.SHAP_VALUES)):
        if col_c.button("Generate AI Explanation"):
            st.session_state[SessionKeys.CURRENT_PAGE] = "xai"
    else:
        if col_d.button("Create Final AI Research Report"):
            st.session_state[SessionKeys.CURRENT_PAGE] = "report"

    # Preserve existing Latest Results section
    if has_dataset() and st.session_state.get(SessionKeys.MODEL_TRAINED):
        st.markdown("---")
        st.subheader("Latest Results")
        best = st.session_state.get(SessionKeys.BEST_MODEL_NAME)
        results = st.session_state.get(SessionKeys.RESULTS) or {}
        if best and best in results:
            score = results[best]
            pt = st.session_state.get(SessionKeys.PROBLEM_TYPE)
            fmt = f"{score:.2%}" if pt == "Classification" else f"{score:.4f}"
            st.info(f"**{best}** — Score: **{fmt}** · Confidence: **{st.session_state.get(SessionKeys.CONFIDENCE_SCORE, 'N/A')}%**")

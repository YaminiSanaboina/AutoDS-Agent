from contextlib import contextmanager
import html

import pandas as pd
import streamlit as st

from config import ACCENT_COLOR, PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR
from utils.session_manager import SessionKeys, get_dataframe, has_dataset


def glass_metric(icon, label, value, delta=None):
    delta_html = f'<div style="color:#10B981;font-size:0.8rem;">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="card-card">
            <div class="card-icon">{icon}</div>
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def glass_panel(title: str | None = None, subtitle: str | None = None):
    html = "<div class='card-panel'>"
    if title:
        html += f"<div class='card-title'>{title}</div>"
    if subtitle:
        html += f"<div class='card-subtitle'>{subtitle}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def glass_panel_small(content: str):
    st.markdown(f"<div class='card-panel-small'>{content}</div>", unsafe_allow_html=True)


def end_glass_panel():
    """Legacy no-op — split HTML wrappers caused raw </div> text in Streamlit."""
    return


def ai_chat_message(message, label="AI Assistant"):
    st.markdown(
        f"""
        <div class="ai-chat-bubble">
            <div class="ai-label">{label}</div>
            <div>{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def issue_badge(text, severity="error"):
    css = {"error": "issue-error", "warn": "issue-warn", "ok": "issue-ok"}.get(severity, "issue-warn")
    color = {"error": "#EF4444", "warn": "#F59E0B", "ok": "#10B981"}.get(severity, "#94A3B8")
    st.markdown(
        f'<span class="issue-badge {css}" style="display:inline-flex;align-items:center;gap:0.4rem;"><span style="width:10px;height:10px;border-radius:50%;background:{color};display:inline-block;"></span>{text}</span>',
        unsafe_allow_html=True,
    )


def rank_medal(rank):
    medals = {1: "1st", 2: "2nd", 3: "3rd"}
    return medals.get(rank, f"#{rank}")


def status_badge(label: str, status: str = "waiting"):
    palette = {
        "completed": ("#10B981", "#ECFDF5"),
        "running": ("#6366F1", "#EEF2FF"),
        "waiting": ("#F59E0B", "#FEF3C7"),
        "error": ("#EF4444", "#FEE2E2"),
    }
    fg, bg = palette.get(status, ("#94A3B8", "#F8FAFC"))
    st.markdown(
        f"<span style='display:inline-flex;align-items:center;gap:0.35rem;padding:0.45rem 0.75rem;border-radius:999px;background:{bg};color:{fg};font-size:0.9rem;font-weight:700;'>{label}</span>",
        unsafe_allow_html=True,
    )


def require_dataset(message="Please upload a dataset from Dataset Hub."):
    """Show gate with navigation to Dataset Hub when no dataset is loaded."""
    if has_dataset():
        return True
    st.info(message)
    if st.button("Upload dataset on Home", key="gate_go_home_upload", type="primary", width="stretch"):
        st.session_state["enterprise_nav_view"] = "home"
        st.rerun()
    return False


def render_no_dataset_gate(message: str = "Please upload a dataset from Dataset Hub") -> bool:
    """Enterprise no-dataset gate — upload exists ONLY in Dataset Hub."""
    return require_dataset(message)


def require_model():
    """Show an informational card when no trained model is available."""
    st.markdown(
        """
        <div class="card-panel">
            <div class="card-title">Train a model in AutoML Studio to unlock predictions and explainability.</div>
            <div style="color:#64748B;font-size:0.95rem;margin-top:0.4rem;">
                Visit <strong>AutoML Studio</strong> to train a model with your dataset.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return False


def require_reports():
    """Show an informational card when reports are not generated yet."""
    st.markdown(
        """
        <div class="card-panel">
            <div class="card-title">Run the Autonomous AI pipeline to generate reports.</div>
            <div style="color:#64748B;font-size:0.95rem;margin-top:0.4rem;">
                Use <strong>Run Analysis</strong> on the Dashboard to produce executive reports.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return False


def dataset_banner():
    df = get_dataframe()
    if df is None:
        return
    name = st.session_state.get(SessionKeys.DATASET_NAME) or "Dataset"
    st.success(f"**{name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")


def metric_card(label: str, value: str, delta: str | None = None):
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def primary_metric_label(problem_type: str | None) -> str:
    """Return the primary metric label for the given problem type.

    - Classification -> "Accuracy"
    - Regression -> "R² Score"
    """
    if problem_type is None:
        return "Accuracy"
    try:
        pt = str(problem_type).lower()
    except Exception:
        return "Accuracy"
    return "R² Score" if "regress" in pt else "Accuracy"


def feature_card(title: str, description: str, icon: str = ""):
    icon_html = f"<span style='font-size:1.5rem;margin-right:0.5rem;'>{icon}</span>" if icon else ""
    st.markdown(
        f"""
        <div class="feature-card">
            <h4>{icon_html}{title}</h4>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def health_badge(text: str, grade_class: str = "health-good"):
    st.markdown(
        f'<span class="health-badge {grade_class}">{text}</span>',
        unsafe_allow_html=True,
    )


def dataset_loaded_banner(df, filename: str | None = None):
    if df is None:
        return
    name = filename or "Dataset"
    st.success(f"**{name}** loaded — {df.shape[0]:,} rows × {df.shape[1]} columns")


def issue_card(title: str, description: str, recommendation: str, severity: str = "medium"):
    css = "critical" if severity in ("high", "critical", "error") else ""
    st.markdown(
        f"""
        <div class="issue-card {css}">
            <strong>{title}</strong>
            <p style="margin:0.35rem 0 0 0;color:#475569;">{description}</p>
            <p style="margin:0.35rem 0 0 0;color:#64748B;font-size:0.9rem;"><em>{recommendation}</em></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_box(title: str, content: str):
    st.markdown(
        f"""
        <div class="insight-box">
            <h5>{title}</h5>
            <p style="margin:0;color:#334155;line-height:1.6;">{content}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ai_assistant_panel(summary: str, details: list[str], recommendation: str | None = None, title: str = "AI Assistant"):
    """
    Render a professional AI assistant panel with collapsible detailed analysis.
    
    Args:
        summary: Short highlighted summary text (always visible)
        details: List of detailed explanation points (shown in collapsible section)
        recommendation: Optional recommended next step text
        title: Panel title (default: "AI Assistant")
    """
    rec_html = (
        f"<div class='card-panel-small'><strong>Recommended Next Step</strong><br>{recommendation}</div>"
        if recommendation
        else ""
    )
    st.markdown(
        f"""
        <div class="card-panel">
            <div class="card-title">{title}</div>
            <div style="color:#475569;font-size:0.98rem;line-height:1.7;margin-bottom:1rem;">
                <strong>Quick Summary</strong><br>{summary}
            </div>
            {rec_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if details:
        with st.expander("Show Detailed Analysis"):
            for detail in details:
                st.markdown(f"- {detail}")


def render_ai_chat_workspace(
    chat_session_key: str,
    generate_response_fn,
    *,
    title: str = "AI Data Scientist Assistant",
    subtitle: str = "Ask about your dataset, models, SHAP, deployment, and reports.",
    suggested_questions: list[str] | None = None,
    chat_input_placeholder: str = "Ask about your dataset or model...",
    pending_key: str | None = None,
    open_panel_fn=None,
    close_panel_fn=None,
):
    """Render a ChatGPT-style assistant with memory, timestamps, and suggested prompts."""
    import datetime

    if open_panel_fn:
        open_panel_fn(title, subtitle)
    else:
        glass_panel(title, subtitle)

    st.session_state.setdefault(chat_session_key, [])
    chat_history = st.session_state[chat_session_key]
    pending_key = pending_key or f"{chat_session_key}_pending"
    suggested = suggested_questions or [
        "What is my dataset about?",
        "Which model performed best?",
        "Which features are important?",
        "Can I deploy this model?",
        "Explain this for a beginner",
    ]

    st.markdown("<div class='chat-suggestions-label'>Suggested questions</div>", unsafe_allow_html=True)
    chip_cols = st.columns(min(len(suggested), 3))
    for idx, question in enumerate(suggested[:6]):
        with chip_cols[idx % len(chip_cols)]:
            if st.button(question, key=f"{chat_session_key}_suggest_{idx}", width="stretch"):
                st.session_state[pending_key] = question

    pending = st.session_state.get(pending_key)
    if pending:
        st.session_state[pending_key] = None
        user_text = pending
        st.session_state[chat_session_key].append({
            "role": "user",
            "message": user_text,
            "timestamp": datetime.datetime.now().strftime("%H:%M"),
        })
        with st.spinner("AI is thinking..."):
            answer = generate_response_fn(user_text, chat_history=st.session_state[chat_session_key])
        st.session_state[chat_session_key].append({
            "role": "assistant",
            "message": answer,
            "timestamp": datetime.datetime.now().strftime("%H:%M"),
        })
        st.rerun()

    if not chat_history:
        st.markdown(
            "<div class='assistant-welcome'>"
            "Hello! I have access to your dataset, pipeline results, models, and reports. "
            "Pick a suggested question or type your own."
            "</div>",
            unsafe_allow_html=True,
        )

    for chat_item in chat_history:
        role = chat_item.get("role")
        message = chat_item.get("message", "")
        timestamp = chat_item.get("timestamp", "")
        from ui.saas_components import render_ai_response_card
        render_ai_response_card(message, role=role or "assistant", timestamp=timestamp)

    user_input = st.chat_input(chat_input_placeholder, key=f"{chat_session_key}_input")
    if user_input:
        st.session_state[chat_session_key].append({
            "role": "user",
            "message": user_input,
            "timestamp": datetime.datetime.now().strftime("%H:%M"),
        })
        with st.spinner("AI is thinking..."):
            answer = generate_response_fn(user_input, chat_history=st.session_state[chat_session_key])
        st.session_state[chat_session_key].append({
            "role": "assistant",
            "message": answer,
            "timestamp": datetime.datetime.now().strftime("%H:%M"),
        })
        st.rerun()

    if close_panel_fn:
        close_panel_fn()
    else:
        end_glass_panel()


# ── Enterprise SaaS Components ─────────────────────────────────────────────

@contextmanager
def enterprise_panel(title: str, subtitle: str | None = None):
    """Premium bordered panel for Streamlit widgets (avoids split HTML div wrappers)."""
    sub = f'<p class="ads-panel-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="ads-panel-heading"><div class="card-title">{title}</div>{sub}</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        yield


def render_autonomous_run_button(*, key: str, disabled: bool, is_running: bool) -> bool:
    """Prominent gradient run button for the full analysis pipeline."""
    label = "⟳ Running Analysis…" if is_running else "🚀 Run Analysis"
    return st.button(
        label,
        key=key,
        disabled=disabled,
        type="primary",
        width="stretch",
        help="Run profiling, EDA, AutoML, SHAP explainability, and AI insights on the loaded dataset.",
    )


def render_panel_header(title: str, subtitle: str | None = None) -> None:
    """Static section header without unclosed HTML wrappers."""
    sub = f'<p class="ads-section-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(f'<h2 class="ads-section-title">{title}</h2>{sub}', unsafe_allow_html=True)


from utils.faculty_labels import FACULTY_HELP, FACULTY_LABELS, NAV_ITEMS

ENTERPRISE_NAV_ITEMS = [
    ("home", "Home"),
    ("reports", "Reports"),
]

ENTERPRISE_NAV_ITEMS_FULL = ENTERPRISE_NAV_ITEMS

VIEW_ALIASES = {
    "dashboard": "home",
    "dataset_hub": "home",
    "eda_lab": "reports",
    "automl_studio": "reports",
    "explainability": "reports",
    "ethics": "reports",
    "ai_cleaning": "reports",
    "deployment": "reports",
    "ai_assistant": "home",
    "image_analysis": "home",
}


def normalize_nav_view(view: str) -> str:
    """Map legacy view keys to Home / Assistant / Reports."""
    return VIEW_ALIASES.get(view, view)


def render_command_bar(
    *,
    version: str = "v2.0",
    production_badge: str,
    production_class: str = "ads-badge-ready",
    smart_mode: bool = True,
    timer_text: str = "00:00",
    session_id: str = "—",
    pipeline_status: str = "Ready",
    dataset_status: str = "Not Loaded",
    current_stage: str = "—",
) -> None:
    """Top gradient command bar with badges and session metadata."""
    smart_html = (
        '<span class="ads-badge ads-badge-smart">⚡ Smart Mode</span>'
        if smart_mode
        else ""
    )
    st.markdown(
        f"""
        <div class="ads-command-bar">
            <div class="ads-command-brand">
                <div class="ads-logo-mark">🤖</div>
                <div class="ads-brand-text">
                    <h2>AutoDS</h2>
                    <p>Autonomous Data Science Platform</p>
                </div>
            </div>
            <div class="ads-command-meta">
                <span class="ads-badge ads-badge-version">AutoDS Enterprise</span>
                <span class="ads-badge {production_class}">{production_badge}</span>
                {smart_html}
                <span class="ads-badge ads-badge-timer">{timer_text}</span>
                <span class="ads-badge ads-badge-agent">{pipeline_status}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_enterprise_sidebar(
    *,
    active_view: str,
    dataset_status: str,
    health_score: str,
    current_stage: str,
    on_nav_key_prefix: str = "ent_nav",
    compact: bool = True,
) -> str:
    """Render glassmorphism left sidebar navigation. Returns selected view key."""
    nav_items = ENTERPRISE_NAV_ITEMS if compact else ENTERPRISE_NAV_ITEMS_FULL
    st.markdown(
        """
        <div class="ads-sidebar-top">
            <div class="ads-sidebar-logo">
                <div class="logo-icon">🤖</div>
                <h3>AutoDS</h3>
                <p>Autonomous AI Scientist</p>
            </div>
            <div class="ads-nav-label">Navigation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = active_view
    for key, label in nav_items:
        is_active = key == active_view
        btn_type = "primary" if is_active else "secondary"
        css = "ads-nav-compact"
        if st.button(label, key=f"{on_nav_key_prefix}_{key}", width="stretch", type=btn_type):
            selected = key

    st.markdown(
        f"""
        <div class="ads-sidebar-footer">
            <div class="ads-status-row"><span>Dataset Status</span><span>{dataset_status}</span></div>
            <div class="ads-status-row"><span>Health Score</span><span class="ads-health-pill">{health_score}</span></div>
            <div class="ads-status-row"><span>Current Stage</span><span>{current_stage}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return selected


def render_enterprise_hero(
    *,
    title: str = "AI Command Center",
    subtitle: str,
    health: str,
    confidence: str,
    best_model: str,
    deployment: str,
) -> None:
    """Large hero card with animated metric counters."""
    st.markdown(
        f"""
        <div class="ads-hero">
            <div class="ads-hero-kicker">Enterprise AI Platform</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <div class="ads-hero-metrics">
                <div class="ads-metric-card">
                    <div class="ads-metric-label">Dataset Health</div>
                    <div class="ads-metric-value">{health}</div>
                    <div class="ads-metric-help">{FACULTY_HELP["dataset_health"]}</div>
                </div>
                <div class="ads-metric-card">
                    <div class="ads-metric-label">Model Reliability</div>
                    <div class="ads-metric-value">{confidence}</div>
                    <div class="ads-metric-help">{FACULTY_HELP["model_reliability"]}</div>
                </div>
                <div class="ads-metric-card">
                    <div class="ads-metric-label">Best Model</div>
                    <div class="ads-metric-value">{best_model}</div>
                </div>
                <div class="ads-metric-card">
                    <div class="ads-metric-label">Production Readiness</div>
                    <div class="ads-metric-value">{deployment}</div>
                    <div class="ads-metric-help">{FACULTY_HELP["production_readiness"]}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero_run_toolbar(*, key: str, disabled: bool, is_running: bool) -> bool:
    """Run control aligned beside the dashboard hero."""
    _, run_col = st.columns([3.4, 1.2])
    with run_col:
        return render_autonomous_run_button(key=key, disabled=disabled, is_running=is_running)
    return False


def render_insight_panel(
    *,
    ai_status: str,
    current_task: str,
    progress_pct: int,
    remaining_time: str,
    best_model: str,
    score: str,
    confidence: str,
    deployment_status: str,
    recommendations: list[str],
) -> None:
    """Fixed right-side AI insight panel."""
    rec_html = "".join(f'<div class="ads-rec-item">{rec}</div>' for rec in recommendations[:5])
    if not rec_html:
        rec_html = '<div class="ads-rec-item">Run the pipeline to receive live AI recommendations.</div>'

    st.markdown(
        f"""
        <div class="ads-insight-panel">
            <div class="ads-insight-card">
                <h4>Agent Status</h4>
                <div class="ads-insight-stat"><span class="label">Status</span><span class="value">{ai_status}</span></div>
                <div class="ads-insight-stat"><span class="label">Current Task</span><span class="value">{current_task}</span></div>
                <div class="ads-progress-bar"><div class="ads-progress-fill" style="width:{min(100, progress_pct)}%;"></div></div>
                <div class="ads-insight-stat"><span class="label">Progress</span><span class="value">{progress_pct}%</span></div>
                <div class="ads-insight-stat"><span class="label">Remaining</span><span class="value">{remaining_time}</span></div>
            </div>
            <div class="ads-insight-card">
                <h4>Model Intelligence</h4>
                <div class="ads-insight-stat"><span class="label">Best Model</span><span class="value">{best_model}</span></div>
                <div class="ads-insight-stat"><span class="label">Score</span><span class="value">{score}</span></div>
                <div class="ads-insight-stat"><span class="label">Model Reliability</span><span class="value">{confidence}</span></div>
                <div class="ads-insight-stat"><span class="label">Production Readiness</span><span class="value">{deployment_status}</span></div>
            </div>
            <div class="ads-insight-card">
                <h4>Live Recommendations</h4>
                {rec_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_agent_terminal(lines: list[tuple[str, str]]) -> None:
    """Terminal-style live agent activity feed. Each line: (text, css_class)."""
    body = "".join(
        f'<div class="ads-terminal-line {css}">{text}</div>'
        for text, css in lines
    )
    if not body:
        body = '<div class="ads-terminal-line pending">Waiting for agent activity…</div>'

    st.markdown(
        f"""
        <div class="ads-glass-card" style="padding:0;overflow:hidden;">
            <div class="ads-terminal">
                <div class="ads-terminal-header">
                    <span class="ads-terminal-dot" style="background:#EF4444;"></span>
                    <span class="ads-terminal-dot" style="background:#F59E0B;"></span>
                    <span class="ads-terminal-dot" style="background:#10B981;"></span>
                    <span style="color:#64748B;font-size:0.72rem;margin-left:8px;">Live Agent Activity</span>
                </div>
                {body}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_leaderboard_table(rows: list[dict], best_model: str | None = None, *, compact: bool = False) -> None:
    """Professional AutoML ranking table rendered as a native Streamlit dataframe."""
    if not rows:
        if not compact:
            st.header("🏆 Model Leaderboard")
        st.info("Train models to populate the leaderboard.")
        return

    display_rows = []
    for row in rows:
        display_rows.append(
            {
                "Rank": row.get("rank", "—"),
                "Model": row.get("model", "—"),
                "Accuracy": row.get("accuracy", "—"),
                "F1": row.get("f1", "—"),
                "Training Time": row.get("time", "—"),
                "Status": row.get("status", ""),
            }
        )

    if not compact:
        st.markdown('<div class="ads-leaderboard-header"><h3>🏆 Model Leaderboard</h3></div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(display_rows), width="stretch", hide_index=True)


def render_report_preview_cards() -> None:
    """Report center preview card grid."""
    st.markdown(
        """
        <div class="ads-preview-grid">
            <div class="ads-preview-card"><div class="icon">📋</div><h5>Executive Summary</h5><p>AI-generated business overview and key findings.</p></div>
            <div class="ads-preview-card"><div class="icon">📊</div><h5>EDA Findings</h5><p>Statistical insights, distributions, and correlations.</p></div>
            <div class="ads-preview-card"><div class="icon">🏆</div><h5>Model Performance</h5><p>Leaderboard, validation metrics, and selection rationale.</p></div>
            <div class="ads-preview-card"><div class="icon">🧠</div><h5>Explainability</h5><p>SHAP feature importance and driver analysis.</p></div>
            <div class="ads-preview-card"><div class="icon">🚀</div><h5>Deployment Readiness</h5><p>Risk assessment and production recommendations.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_executive_banner(label: str, banner_class: str = "") -> None:
    """Production-ready executive banner."""
    st.markdown(
        f'<div class="ads-executive-banner {banner_class}"><h3>{label}</h3></div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "") -> None:
    """Workspace section header."""
    sub = f'<p class="ads-section-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(f'<h2 class="ads-section-title">{title}</h2>{sub}', unsafe_allow_html=True)

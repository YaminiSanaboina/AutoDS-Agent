import streamlit as st

from config import (
    DANGER_COLOR,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    SUCCESS_COLOR,
    WARNING_COLOR,
)


def inject_global_styles():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        }}

        [data-testid="stSidebar"] * {{
            color: #E2E8F0 !important;
        }}

        [data-testid="stSidebar"] .stRadio label {{
            background: transparent;
            border-radius: 8px;
            padding: 0.35rem 0.5rem;
            transition: background 0.2s;
        }}

        [data-testid="stSidebar"] .stRadio label:hover {{
            background: rgba(99, 102, 241, 0.15);
        }}

        .platform-header {{
            background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%);
            padding: 1.75rem 2rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 40px rgba(99, 102, 241, 0.25);
        }}

        .platform-header h1 {{
            margin: 0;
            font-size: 1.75rem;
            font-weight: 700;
        }}

        .platform-header p {{
            margin: 0.35rem 0 0 0;
            opacity: 0.92;
            font-size: 1rem;
        }}

        .metric-card {{
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 4px 20px rgba(15, 23, 42, 0.06);
            height: 100%;
        }}

        .metric-card .label {{
            color: #64748B;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .metric-card .value {{
            color: #0F172A;
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0.35rem 0;
        }}

        .metric-card .delta {{
            color: {SUCCESS_COLOR};
            font-size: 0.85rem;
        }}

        .feature-card {{
            background: linear-gradient(145deg, #FFFFFF 0%, #F8FAFC 100%);
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.25rem;
            height: 100%;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .feature-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(99, 102, 241, 0.12);
        }}

        .feature-card h4 {{
            color: #0F172A;
            margin: 0 0 0.5rem 0;
            font-size: 1rem;
        }}

        .feature-card p {{
            color: #64748B;
            margin: 0;
            font-size: 0.875rem;
            line-height: 1.5;
        }}

        .health-badge {{
            display: inline-block;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.85rem;
        }}

        .health-excellent {{ background: #D1FAE5; color: #065F46; }}
        .health-good {{ background: #DBEAFE; color: #1E40AF; }}
        .health-fair {{ background: #FEF3C7; color: #92400E; }}
        .health-poor {{ background: #FEE2E2; color: #991B1B; }}

        .issue-card {{
            border-left: 4px solid {WARNING_COLOR};
            background: #FFFBEB;
            padding: 1rem 1.25rem;
            border-radius: 0 10px 10px 0;
            margin-bottom: 0.75rem;
        }}

        .issue-card.critical {{
            border-left-color: {DANGER_COLOR};
            background: #FEF2F2;
        }}

        .insight-box {{
            background: linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 100%);
            border: 1px solid #C7D2FE;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin: 0.75rem 0;
        }}

        .insight-box h5 {{
            color: {PRIMARY_COLOR};
            margin: 0 0 0.5rem 0;
            font-size: 0.9rem;
        }}

        .section-divider {{
            border: none;
            border-top: 1px solid #E2E8F0;
            margin: 1.5rem 0;
        }}

        div[data-testid="stMetric"] {{
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            padding: 0.5rem 1.25rem;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="platform-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

import streamlit as st

from pathlib import Path

from config import (
    ACCENT_COLOR,
    NAV_ITEMS,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
)
from utils.session_manager import SessionKeys, init_session

_ASSETS_CSS = Path(__file__).resolve().parents[1] / "assets" / "style.css"


def inject_styles():
    css_extra = ""
    if _ASSETS_CSS.exists():
        css_extra = _ASSETS_CSS.read_text(encoding="utf-8")

    st.markdown(
        f"""
        <style>
        {css_extra}
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        *, *::before, *::after {{ box-sizing: border-box; }}
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: #0F172A !important;
            background: #F5F7FA !important;
        }}

        .block-container {{
            padding-top: 1.25rem;
            padding-bottom: 1.25rem;
            max-width: 1580px;
            width: 100%;
            background: transparent !important;
        }}

        .stApp {{
            background: #F5F7FA !important;
        }}

        .stSidebar {{
            background: #FFFFFF !important;
            color: #0F172A !important;
            border-right: 1px solid rgba(148,163,184,0.18) !important;
            box-shadow: 0 18px 50px rgba(15,23,42,0.06) !important;
        }}

        .sidebar-section {{
            padding: 0.85rem 0 0.35rem;
            color: #475569;
            font-size: 0.9rem;
            font-weight: 700;
        }}

        .sidebar-title {{
            font-size: 1.1rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.35rem;
        }}

        .sidebar-subtitle {{
            color: #64748B;
            font-size: 0.9rem;
            line-height: 1.6;
            margin-bottom: 1rem;
        }}

        .nav-button {{
            width: 100%;
            text-align: left;
            padding: 0.85rem 1rem;
            border-radius: 14px;
            background: #F8FAFC;
            color: #0F172A;
            border: 1px solid rgba(148,163,184,0.28);
            margin-bottom: 0.6rem;
            font-weight: 700;
            transition: all 180ms ease;
        }}

        .nav-button:hover {{
            background: #FFFFFF;
            color: #0F172A;
            transform: translateX(1px);
        }}

        .nav-button.active {{
            background: linear-gradient(135deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
            border-color: transparent;
            color: #FFFFFF;
            box-shadow: 0 12px 28px rgba(99,102,241,0.18);
        }}

        .hero-section {{
            background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(238,242,255,0.92));
            border-radius: 22px;
            padding: 1.6rem 1.75rem;
            color: #0F172A;
            margin-bottom: 1.4rem;
            box-shadow: 0 22px 60px rgba(15, 23, 42, 0.08);
            border: 1px solid rgba(148,163,184,0.16);
        }}

        .premium-hero {{
            background: linear-gradient(135deg, #FFFFFF 0%, #EEF2FF 48%, #F5F3FF 100%);
            position: relative;
            overflow: hidden;
        }}

        .hero-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.45rem 0.85rem;
            border-radius: 999px;
            background: rgba(99,102,241,0.12);
            color: {PRIMARY_COLOR};
            font-weight: 700;
            font-size: 0.88rem;
            margin-bottom: 0.85rem;
        }}

        .hero-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin-top: 1.25rem;
        }}

        .hero-metric-card {{
            background: rgba(255,255,255,0.72);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 14px 34px rgba(99,102,241,0.08);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }}

        .hero-metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 18px 40px rgba(99,102,241,0.14);
        }}

        .hero-metric-label {{
            color: #64748B;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .hero-metric-value {{
            margin-top: 0.45rem;
            font-size: 1.45rem;
            font-weight: 800;
            color: #0F172A;
        }}

        .timeline-vertical {{
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
            margin: 16px 0;
        }}

        .timeline-row {{
            display: grid;
            grid-template-columns: 42px 1fr;
            gap: 12px;
            align-items: start;
        }}

        .timeline-marker {{
            width: 42px;
            height: 42px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #E2E8F0;
            color: #475569;
            font-weight: 800;
            margin-top: 0.85rem;
            box-shadow: inset 0 0 0 2px rgba(255,255,255,0.8);
        }}

        .timeline-row.done .timeline-marker {{
            background: rgba(16,185,129,0.18);
            color: #059669;
        }}

        .timeline-row.active .timeline-marker {{
            background: rgba(99,102,241,0.16);
            color: {PRIMARY_COLOR};
            animation: pulseGlow 1.6s ease-in-out infinite;
        }}

        @keyframes pulseGlow {{
            0%, 100% {{ box-shadow: 0 0 0 0 rgba(99,102,241,0.25); }}
            50% {{ box-shadow: 0 0 0 10px rgba(99,102,241,0); }}
        }}

        @media (prefers-color-scheme: dark) {{
            html, body, [class*="css"] {{
                color: #E2E8F0 !important;
                background: #0B1220 !important;
            }}
            .stApp {{
                background: radial-gradient(circle at top, #111827 0%, #0B1220 55%) !important;
            }}
            .hero-section, .glass-panel, .card-panel, .timeline-card, .hero-metric-card {{
                background: rgba(15,23,42,0.82) !important;
                color: #E2E8F0 !important;
                border-color: rgba(148,163,184,0.22) !important;
            }}
            .hero-section h1, .card-title, .hero-metric-value {{
                color: #F8FAFC !important;
            }}
            .hero-section p, .card-subtitle, .hero-metric-label {{
                color: #94A3B8 !important;
            }}
        }}

        .hero-section h1 {{
            margin: 0;
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.05;
        }}

        .hero-section p {{
            margin: 0.8rem 0 0;
            color: #475569;
            font-size: 1rem;
            line-height: 1.7;
        }}

        .glass-panel, .card-panel {{
            background: #FFFFFF;
            border-radius: 22px;
            padding: 1.35rem 1.35rem;
            border: 1px solid rgba(148,163,184,0.18);
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.35rem;
        }}

        .glass-panel-small, .card-panel-small {{
            background: #F8FAFC;
            border-radius: 18px;
            padding: 1rem;
            border: 1px solid rgba(148,163,184,0.18);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }}

        .glass-title, .card-title {{
            font-size: 1.12rem;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 0.65rem;
        }}

        .glass-subtitle, .card-subtitle {{
            color: #475569;
            font-size: 0.95rem;
            line-height: 1.7;
            margin-bottom: 1rem;
        }}

        .glass-card, .card-card {{
            background: #FFFFFF;
            border: 1px solid rgba(148,163,184,0.16);
            border-radius: 18px;
            padding: 1rem;
            color: #0F172A;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            height: 100%;
        }}

        .glass-card:hover, .card-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 18px 42px rgba(15, 23, 42, 0.12);
        }}

        .glass-card .card-icon, .card-card .card-icon {{
            font-size: 1.5rem;
            margin-bottom: 0.45rem;
            color: {PRIMARY_COLOR};
        }}
        .glass-card .card-label, .card-card .card-label {{
            color: #475569;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }}
        .glass-card .card-value, .card-card .card-value {{
            color: #0F172A;
            font-size: 1.55rem;
            font-weight: 800;
            margin: 0;
        }}

        .workflow-step {{
            background: #F8FAFC;
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 16px;
            padding: 0.85rem 1rem;
            text-align: center;
            font-weight: 700;
            font-size: 0.95rem;
            color: #0F172A;
            box-shadow: 0 8px 20px rgba(15,23,42,0.06);
            transition: all 0.18s ease;
        }}

        .workflow-step.active {{
            background: linear-gradient(135deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
            color: #FFFFFF;
            border-color: transparent;
            transform: scale(1.01);
            box-shadow: 0 12px 26px rgba(99,102,241,0.14);
        }}

        .workflow-step.done {{
            background: rgba(16,185,129,0.12);
            border-color: rgba(16,185,129,0.30);
            color: #065F46;
        }}

        .timeline-card {{
            background: #FFFFFF;
            border: 1px solid rgba(148,163,184,0.20);
            border-radius: 20px;
            padding: 1rem 1.15rem;
            min-height: 138px;
            box-shadow: 0 14px 35px rgba(15,23,42,0.06);
            transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
        }}

        .timeline-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(99,102,241,0.25);
            box-shadow: 0 16px 36px rgba(15,23,42,0.08);
        }}

        .timeline-card.active {{
            border-color: {SECONDARY_COLOR};
            box-shadow: 0 18px 40px rgba(99,102,241,0.14);
        }}

        .timeline-card.done {{
            background: rgba(16,185,129,0.08);
            border-color: rgba(16,185,129,0.30);
        }}

        .step-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            color: #FFFFFF;
            background: {SECONDARY_COLOR};
        }}

        .workflow-arrow {{
            text-align: center;
            color: {PRIMARY_COLOR};
            font-size: 1.1rem;
            font-weight: 700;
        }}

        .ai-chat-bubble, .assistant-bubble {{
            background: #FFFFFF;
            border: 1px solid rgba(148,163,184,0.20);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin: 0.35rem 0;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }}

        .ai-chat-bubble .ai-label, .assistant-bubble .ai-label {{
            color: {PRIMARY_COLOR};
            font-weight: 700;
            font-size: 0.92rem;
            margin-bottom: 0.45rem;
        }}

        .chat-timestamp {{
            display: block;
            color: #94A3B8;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }}

        .assistant-welcome {{
            background: rgba(99,102,241,0.08);
            border: 1px solid rgba(99,102,241,0.18);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            color: #475569;
            margin-bottom: 1rem;
            line-height: 1.6;
        }}

        .chat-suggestions-label {{
            color: #64748B;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin: 0.5rem 0 0.65rem;
        }}

        .issue-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.4rem 0.8rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 700;
            margin: 0.15rem;
        }}
        .issue-error {{ background: #FEE2E2; color: #991B1B; }}
        .issue-warn {{ background: #FEF3C7; color: #92400E; }}
        .issue-ok {{ background: #D1FAE5; color: #065F46; }}

        .rank-gold {{ color: #D97706; font-weight: 800; }}
        .rank-silver {{ color: #6B7280; font-weight: 800; }}
        .rank-bronze {{ color: #B45309; font-weight: 800; }}

        .resource-card {{
            background: #FFFFFF;
            border: 1px solid rgba(148,163,184,0.22);
            border-radius: 18px;
            padding: 0.95rem;
            height: 100%;
            transition: all 0.20s ease;
        }}
        .resource-card:hover {{
            border-color: {PRIMARY_COLOR};
            box-shadow: 0 10px 30px rgba(99,102,241,0.12);
            transform: translateY(-2px);
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(12px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateX(-14px); }}
            to {{ opacity: 1; transform: translateX(0); }}
        }}

        h1 {{ margin-top: 0.8rem !important; margin-bottom: 0.5rem !important; }}
        h2 {{ margin-top: 0.6rem !important; margin-bottom: 0.4rem !important; }}
        h3 {{ margin-top: 0.5rem !important; margin-bottom: 0.3rem !important; }}

        div[data-testid="stExpander"] {{
            border: 1px solid rgba(148,163,184,0.24) !important;
            border-radius: 14px !important;
            background: #FFFFFF !important;
        }}

        .stCaption {{
            margin: 0.3rem 0 !important;
            font-size: 0.85rem !important;
            color: #475569 !important;
        }}

        div[data-testid="stMetric"] {{
            background: #FFFFFF;
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 14px;
            padding: 0.5rem 0.75rem;
        }}

        .stTabs [data-baseweb="tab"] {{
            background: #F8FAFC !important;
            border-radius: 12px 12px 0 0 !important;
            color: #0F172A !important;
            font-size: 0.95rem;
            font-weight: 700;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            background: #FFFFFF !important;
        }}

        .stDataFrame {{
            background: #FFFFFF !important;
            border-radius: 18px !important;
            padding: 1rem !important;
        }}

        .stButton>button {{
            border-radius: 14px !important;
            padding: 0.9rem 1rem !important;
            font-weight: 700 !important;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08) !important;
            transition: transform 0.2s ease !important;
        }}
        .stButton>button:hover {{
            transform: translateY(-1px) !important;
        }}
        .stSidebar .stButton>button {{
            background: #F8FAFC !important;
            color: #0F172A !important;
            border: 1px solid rgba(148,163,184,0.18) !important;
        }}
        .stSidebar .stButton>button:hover {{
            background: #FFFFFF !important;
            color: #0F172A !important;
        }}
        .stSidebar .stButton>button[kind="primary"] {{
            background: linear-gradient(135deg, {PRIMARY_COLOR}, {SECONDARY_COLOR}) !important;
            color: #FFFFFF !important;
            border-color: transparent !important;
        }}
        .stSidebar .stButton>button[kind="secondary"] {{
            background: #F8FAFC !important;
            color: #0F172A !important;
            border: 1px solid rgba(148,163,184,0.18) !important;
        }}
        .stButton>button[kind="primary"] {{
            background: linear-gradient(135deg, {PRIMARY_COLOR}, {ACCENT_COLOR}) !important;
            color: white !important;
            border: none !important;
        }}
        .stButton>button[kind="secondary"] {{
            background: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid rgba(148,163,184,0.25) !important;
        }}

        .stTextInput>div>div>input,
        .stSelectbox>div>div>div>div,
        .stNumberInput>div>div>input {{
            border-radius: 14px !important;
            border: 1px solid rgba(148,163,184,0.28) !important;
            background: #FFFFFF !important;
            color: #0F172A !important;
        }}

        .stFileUploader>div>div {{
            background: #FFFFFF !important;
            border: 1px dashed rgba(99,102,241,0.35) !important;
            border-radius: 16px !important;
        }}

        .stAlert {{
            border-radius: 18px !important;
            border: 1px solid rgba(148,163,184,0.16) !important;
            background: #F8FAFC !important;
            color: #0F172A !important;
            box-shadow: 0 16px 30px rgba(15,23,42,0.08) !important;
        }}
        .stInfo > div[data-testid="stMarkdownContainer"] p,
        .stSuccess > div[data-testid="stMarkdownContainer"] p,
        .stError > div[data-testid="stMarkdownContainer"] p {{
            color: #0F172A !important;
        }}
        .stInfo {{
            border-color: rgba(56,189,253,0.35) !important;
            background: rgba(56,189,253,0.12) !important;
        }}
        .stSuccess {{
            border-color: rgba(16,185,129,0.35) !important;
            background: rgba(16,185,129,0.12) !important;
        }}
        .stError {{
            border-color: rgba(239,68,68,0.35) !important;
            background: rgba(239,68,68,0.12) !important;
        }}

        .chief-decision-shell {{
            position: relative;
            margin: 1.5rem 0 2rem;
            padding: 1.75rem;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(248,250,252,0.88));
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid rgba(148,163,184,0.22);
            box-shadow: 0 28px 60px rgba(15,23,42,0.10);
        }}
        .chief-decision-shell::before {{
            content: "";
            position: absolute;
            inset: -1px;
            border-radius: 29px;
            padding: 1px;
            background: linear-gradient(135deg, {PRIMARY_COLOR}, {ACCENT_COLOR}, {SECONDARY_COLOR});
            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            pointer-events: none;
        }}
        .chief-decision-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.25rem;
        }}
        .chief-decision-icon {{
            width: 56px;
            height: 56px;
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.75rem;
            background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(56,189,253,0.18));
            border: 1px solid rgba(99,102,241,0.18);
        }}
        .chief-decision-kicker {{
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #64748B;
        }}
        .chief-decision-title {{
            font-size: 1.65rem;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.2;
        }}
        .chief-decision-card {{
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 20px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 0.85rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.65);
        }}
        .chief-section-label {{
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #64748B;
            margin-bottom: 0.55rem;
        }}
        .chief-model-name {{
            font-size: 1.45rem;
            font-weight: 800;
            color: #0F172A;
            margin-bottom: 0.65rem;
        }}
        .chief-model-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem 1.25rem;
            color: #475569;
            font-size: 0.95rem;
        }}
        .chief-model-meta strong {{
            color: #0F172A;
        }}
        .chief-summary {{
            background: rgba(99,102,241,0.08);
            border-left: 4px solid {PRIMARY_COLOR};
            border-radius: 0 16px 16px 0;
            padding: 0.95rem 1.1rem;
            color: #334155;
            margin: 0.75rem 0 1rem;
            line-height: 1.65;
        }}
        .chief-stat-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 0.75rem;
        }}
        .chief-stat {{
            min-width: 90px;
            color: #64748B;
            font-size: 0.82rem;
            line-height: 1.5;
        }}
        .chief-stat strong {{
            display: block;
            color: #0F172A;
            font-size: 1.15rem;
        }}
        .chief-muted-text {{
            color: #64748B;
            font-size: 0.92rem;
            line-height: 1.6;
        }}
        .chief-strength {{
            background: rgba(16,185,129,0.10);
            border: 1px solid rgba(16,185,129,0.22);
            color: #065F46;
            border-radius: 14px;
            padding: 0.65rem 0.85rem;
            margin-bottom: 0.55rem;
            font-weight: 600;
        }}
        .chief-risk-badge {{
            display: inline-block;
            background: rgba(245,158,11,0.12);
            border: 1px solid rgba(245,158,11,0.28);
            color: #92400E;
            border-radius: 999px;
            padding: 0.45rem 0.85rem;
            margin: 0 0.45rem 0.55rem 0;
            font-size: 0.88rem;
            font-weight: 600;
        }}
        .chief-business {{
            margin-top: 0.75rem;
        }}
        .chief-business-text {{
            color: #334155;
            font-size: 1rem;
            line-height: 1.75;
        }}
        .chief-deployment-banner {{
            margin-top: 1rem;
            padding: 1.1rem 1.2rem;
            border-radius: 20px;
            background: rgba(255,255,255,0.75);
            border: 2px solid rgba(148,163,184,0.22);
        }}
        .chief-deployment-status {{
            font-size: 1.35rem;
            font-weight: 800;
            margin: 0.35rem 0 0.5rem;
        }}
        .chief-muted {{
            color: #64748B;
            font-size: 0.95rem;
        }}

        .report-center-shell {{
            position: relative;
            margin: 1.5rem 0 2rem;
            padding: 1.75rem;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(255,255,255,0.94), rgba(248,250,252,0.90));
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid rgba(148,163,184,0.22);
            box-shadow: 0 28px 60px rgba(15,23,42,0.10);
            animation: reportFadeIn 0.45s ease;
        }}
        @keyframes reportFadeIn {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .report-center-shell::before {{
            content: "";
            position: absolute;
            inset: -1px;
            border-radius: 29px;
            padding: 1px;
            background: linear-gradient(135deg, {PRIMARY_COLOR}, {ACCENT_COLOR}, {SECONDARY_COLOR});
            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            pointer-events: none;
        }}
        .report-center-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.25rem;
        }}
        .report-center-icon {{
            width: 56px;
            height: 56px;
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.75rem;
            background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(56,189,253,0.18));
            border: 1px solid rgba(99,102,241,0.18);
        }}
        .report-center-kicker {{
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #64748B;
        }}
        .report-center-title {{
            font-size: 1.65rem;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.2;
        }}
        .report-kpi-card {{
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(148,163,184,0.18);
            border-left: 4px solid var(--kpi-accent, {PRIMARY_COLOR});
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 12px 28px rgba(15,23,42,0.06);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .report-kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 16px 34px rgba(15,23,42,0.10);
        }}
        .report-kpi-label {{
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #64748B;
            margin-bottom: 0.35rem;
        }}
        .report-kpi-value {{
            font-size: 1.25rem;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.3;
            word-break: break-word;
        }}
        .report-insight-box {{
            background: rgba(99,102,241,0.08);
            border-left: 4px solid {PRIMARY_COLOR};
            border-radius: 0 16px 16px 0;
            padding: 0.95rem 1.1rem;
            color: #334155;
            margin: 0.75rem 0 1rem;
            line-height: 1.65;
        }}
        .report-health-banner {{
            border-radius: 16px;
            padding: 1rem 1.15rem;
            margin: 0.75rem 0 1rem;
            border: 1px solid rgba(148,163,184,0.18);
        }}
        .report-health-banner.health-excellent {{ background: rgba(16,185,129,0.10); }}
        .report-health-banner.health-good {{ background: rgba(59,130,246,0.10); }}
        .report-health-banner.health-fair {{ background: rgba(245,158,11,0.10); }}
        .report-health-banner.health-poor {{ background: rgba(239,68,68,0.10); }}
        .report-muted {{
            color: #64748B;
            font-size: 0.92rem;
            margin-top: 0.35rem;
        }}
        @media (max-width: 768px) {{
            .report-center-shell {{
                padding: 1.1rem;
            }}
            .report-kpi-value {{
                font-size: 1.05rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    init_session()

    current = st.session_state.get(
        SessionKeys.CURRENT_PAGE,
        "ai_command_center"
    )

    st.sidebar.markdown(
        f"""
        <div style='padding:1rem 1rem 0.75rem; background:#F8FAFC; border-radius:18px; border:1px solid rgba(148,163,184,0.2); box-shadow:0 14px 30px rgba(15,23,42,0.06); margin-bottom:1rem;'>
            <div style='font-size:1.3rem; font-weight:800; color:#0F172A;'>AutoDS Agent</div>
            <div style='color:#475569; margin-top:0.35rem; font-size:0.95rem;'>AI workflow platform for data science teams.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # During migration show only AI Command Center in the sidebar navigation.
    st.sidebar.markdown("<div class='sidebar-section'>Navigation</div>", unsafe_allow_html=True)
    active = current == "ai_command_center"
    button_type = "primary" if active else "secondary"
    if st.sidebar.button(
        "⚡ AI Command Center",
        key=f"nav_ai_command_center",
        width="stretch",
        type=button_type,
    ):
        st.session_state[SessionKeys.CURRENT_PAGE] = "ai_command_center"

    st.sidebar.caption("AI Command Center is the single entry point during UI migration.")


def render_hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero-section">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

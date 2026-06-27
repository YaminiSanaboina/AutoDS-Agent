import streamlit as st

from agents.llm_agent import LLMAgent
from utils.context_builder import build_context
from utils.styles import render_hero
from ui.components import render_ai_chat_workspace

SUGGESTED_QUESTIONS = [
    "What is my dataset about?",
    "Why was this model selected?",
    "What are the top features?",
    "What risks exist?",
    "Can this model be deployed?",
    "Summarize the report.",
    "How good is the dataset?",
]


def render():
    render_hero(
        "AI Data Scientist Assistant",
        "Ask questions about your dataset, models, insights, and machine learning workflow.",
    )

    render_ai_chat_workspace(
        "ai_assistant_chat",
        generate_response,
        title="AI Chat Workspace",
        subtitle="Context-aware answers from your full AutoDS project state.",
        suggested_questions=SUGGESTED_QUESTIONS,
        chat_input_placeholder="Ask about dataset quality, models, SHAP, deployment...",
    )


def generate_response(question: str, chat_history: list | None = None) -> str:
    text = (question or "").strip()
    if not text:
        return "Please ask a question about your dataset, model, or analysis so I can help."

    st.session_state.setdefault("llm_agent", LLMAgent())
    llm_agent = st.session_state.get("llm_agent")
    context = build_context()
    try:
        return llm_agent.generate_response(text, context, chat_history=chat_history)
    except Exception:
        return llm_agent._fallback_response(text, context)

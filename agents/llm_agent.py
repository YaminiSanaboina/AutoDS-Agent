"""LLM-backed AI Data Scientist agent for AutoDS.

Supports multiple LLM providers with intelligent local fallback.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import config
from utils.assistant_fallback import build_intelligent_fallback


class LLMAgent:
    """Context-aware AI Data Scientist agent supporting multiple providers."""

    def __init__(self, provider: Optional[str] = None) -> None:
        configured = (provider or getattr(config, "LLM_PROVIDER", "auto") or "auto").strip().lower()
        self.provider = self._resolve_provider(configured)
        self.model_name = self._resolve_model_name()
        self.temperature = getattr(config, "LLM_TEMPERATURE", 0.7)
        self.max_tokens = getattr(config, "LLM_MAX_TOKENS", 512)
        self.timeout = getattr(config, "LLM_API_TIMEOUT", 30)

    def _resolve_provider(self, configured: str) -> str:
        if configured == "fallback":
            return "fallback"
        if configured and configured not in {"auto", "fallback"}:
            return configured
        if getattr(config, "GEMINI_API_KEY", None):
            return "gemini"
        if getattr(config, "GROQ_API_KEY", None):
            return "groq"
        if getattr(config, "OPENAI_API_KEY", None):
            return "openai"
        if getattr(config, "OLLAMA_API_URL", None):
            return "ollama"
        return "fallback"

    def generate_response(
        self,
        user_query: str,
        context: Dict[str, Any],
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate a response using an LLM provider or local intelligent fallback."""
        user_query = (user_query or "").strip()
        if not user_query:
            return "Please ask a question about the dataset, model, or analysis so I can help."

        if self.provider == "fallback":
            return self._fallback_response(user_query, context)

        system_prompt = self.build_system_prompt()
        context_text = self._format_context(context)
        history_text = self._format_history(chat_history)
        user_payload = f"{context_text}\n\n{history_text}\n\nUser question: {user_query}\n\nAnswer in clear, simple language."

        try:
            if self.provider == "openai":
                return self._call_openai(system_prompt, user_payload)
            if self.provider == "gemini":
                return self._call_gemini(system_prompt, user_payload)
            if self.provider == "groq":
                return self._call_groq(system_prompt, user_payload)
            if self.provider == "ollama":
                return self._call_ollama(system_prompt, user_payload)
        except Exception as exc:
            return self._fallback_response(user_query, context, error=exc)

        return self._fallback_response(user_query, context)

    def build_system_prompt(self) -> str:
        return (
            "You are an experienced AI Data Scientist assistant inside AutoDS Agent. "
            "You have access to the user's current dataset, pipeline results, models, SHAP, ethics, and deployment readiness. "
            "Explain technical concepts in simple language. "
            "Answer using only the provided project context when possible. "
            "Give practical suggestions, mention limitations, and highlight risks when relevant."
        )

    def _resolve_model_name(self) -> str:
        if self.provider == "openai":
            return getattr(config, "OPENAI_MODEL", "gpt-4.1-mini")
        if self.provider == "gemini":
            return getattr(config, "GEMINI_MODEL", "gemini-1.5-flash")
        if self.provider == "groq":
            return getattr(config, "GROQ_MODEL", "llama-3.1-8b-instant")
        if self.provider == "ollama":
            return getattr(config, "OLLAMA_MODEL", "llama2")
        return ""

    def _format_history(self, chat_history: Optional[List[Dict[str, str]]]) -> str:
        if not chat_history:
            return ""
        lines = ["Recent conversation:"]
        for item in chat_history[-6:]:
            role = item.get("role", "user")
            message = (item.get("message") or "").strip()
            if message:
                label = "User" if role == "user" else "Assistant"
                lines.append(f"{label}: {message}")
        return "\n".join(lines)

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Serialize rich project context for LLM prompts."""
        try:
            compact = {
                "dataset": context.get("dataset"),
                "model": context.get("model"),
                "eda": context.get("eda"),
                "feature_importance": context.get("feature_importance"),
                "pipeline": context.get("pipeline"),
                "cleaning": context.get("cleaning"),
                "feature_engineering": context.get("feature_engineering"),
                "ethics": context.get("ethics"),
                "deployment": context.get("deployment"),
                "report": {
                    "generated": (context.get("report") or {}).get("generated"),
                    "executive_summary": (context.get("report") or {}).get("executive_summary"),
                },
                "documentation": context.get("documentation"),
            }
            return "AutoDS project context (JSON):\n" + json.dumps(compact, default=str, indent=2)
        except Exception:
            return "AutoDS project context is available but could not be fully serialized."

    def _call_openai(self, system_prompt: str, user_payload: str) -> str:
        if not getattr(config, "OPENAI_API_KEY", None):
            raise RuntimeError("OpenAI API key is not configured.")

        try:
            from openai import OpenAI
        except ImportError:
            import openai  # type: ignore

            openai.api_key = config.OPENAI_API_KEY
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                request_timeout=self.timeout,
            )
            return response.choices[0].message.content.strip()

        client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=self.timeout)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _call_gemini(self, system_prompt: str, user_payload: str) -> str:
        api_key = getattr(config, "GEMINI_API_KEY", None)
        if not api_key:
            raise RuntimeError("Gemini API key is not configured.")

        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Requests package is required for Gemini support.") from exc

        model = self.model_name
        if not model.startswith("models/"):
            model = f"models/{model}"
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={api_key}"
        body = {
            "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_payload}"}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        response = requests.post(endpoint, json=body, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get("candidates") or []
        if candidates:
            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []
            if parts:
                return str(parts[0].get("text", "")).strip()
        return str(payload).strip()

    def _call_groq(self, system_prompt: str, user_payload: str) -> str:
        api_key = getattr(config, "GROQ_API_KEY", None)
        if not api_key:
            raise RuntimeError("Groq API key is not configured.")

        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Requests package is required for Groq support.") from exc

        endpoint = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = requests.post(endpoint, headers=headers, json=body, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        if choices:
            return str(choices[0].get("message", {}).get("content", "")).strip()
        return str(payload).strip()

    def _call_ollama(self, system_prompt: str, user_payload: str) -> str:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Requests package is required for Ollama support.") from exc

        api_url = getattr(config, "OLLAMA_API_URL", "http://localhost:11434").rstrip("/")
        if not api_url:
            raise RuntimeError("Ollama API URL is not configured.")

        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            "stream": False,
        }
        response = requests.post(f"{api_url}/api/chat", json=body, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        message = payload.get("message") or {}
        if message.get("content"):
            return str(message["content"]).strip()
        return str(payload).strip()

    def _fallback_response(
        self,
        user_query: str,
        context: Dict[str, Any],
        error: Optional[Exception] = None,
    ) -> str:
        return build_intelligent_fallback(user_query, context, error=error)

import os

APP_TITLE = "AutoDS Agent"
APP_ICON = ""
APP_TAGLINE = "AI-Powered Autonomous Data Scientist"

NAV_ITEMS = [
    ("ai_command_center", "AI Command Center", ""),
]

PRIMARY_COLOR = "#6366F1"
SECONDARY_COLOR = "#8B5CF6"
ACCENT_COLOR = "#06B6D4"
SUCCESS_COLOR = "#10B981"
WARNING_COLOR = "#F59E0B"
DANGER_COLOR = "#EF4444"
DARK_BG = "#0F172A"

CLASSIFICATION_MODELS = [
    "Random Forest",
    "Logistic Regression",
    "Decision Tree",
    "XGBoost",
    "SVM",
]

REGRESSION_MODELS = [
    "Linear Regression",
    "Random Forest Regressor",
    "XGBoost Regressor",
]

# LLM configuration
LLM_PROVIDER = os.getenv("AUTODS_LLM_PROVIDER", "fallback")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
LLM_TEMPERATURE = float(os.getenv("AUTODS_LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("AUTODS_LLM_MAX_TOKENS", "512"))
LLM_API_TIMEOUT = int(os.getenv("AUTODS_LLM_API_TIMEOUT", "30"))

# Smart Autonomous Mode defaults
SMART_MODE_DEFAULT = True
# Global execution budget (seconds) for the end-to-end pipeline when Smart Mode is enabled
SMART_MODE_BUDGET_SECONDS = int(os.getenv("AUTODS_SMART_BUDGET_SECONDS", "300"))
# Limit iterations for hyperparameter tuning when in Smart Mode
# Limit iterations for hyperparameter tuning when in Smart Mode
# Default to 2 for quick Smart Mode runs
SMART_MODE_HPO_MAX_ITER = int(os.getenv("AUTODS_SMART_HPO_MAX_ITER", "2"))
# Maximum models to try in parallel during Smart Mode (and overall candidates limit)
SMART_MODE_MAX_MODEL_JOBS = int(os.getenv("AUTODS_SMART_MAX_MODEL_JOBS", "15"))
# SHAP sampling size to keep explainability quick
SMART_MODE_SHAP_MAX_SAMPLES = int(os.getenv("AUTODS_SMART_SHAP_MAX_SAMPLES", "100"))
# Disable SHAP explainability in Smart Mode for larger datasets to save time
SMART_MODE_SHAP_DISABLE_ROWS = int(os.getenv("AUTODS_SMART_SHAP_DISABLE_ROWS", "500"))
# Skip hyperparameter optimization in Smart Mode on smaller datasets
SMART_MODE_HPO_SKIP_ROWS = int(os.getenv("AUTODS_SMART_HPO_SKIP_ROWS", "1000"))
# Full Smart Mode candidate sets — train complete registry (optional env override as comma-separated names)
SMART_MODE_MODEL_CANDIDATES = [
    name.strip()
    for name in os.getenv("AUTODS_SMART_MODEL_CANDIDATES", "").split(",")
    if name.strip()
]
SMART_MODE_REGRESSION_CANDIDATES = [
    name.strip()
    for name in os.getenv("AUTODS_SMART_REGRESSION_CANDIDATES", "").split(",")
    if name.strip()
]
# Cache folder for pipeline artifacts
SMART_MODE_CACHE_DIR = os.getenv("AUTODS_SMART_CACHE_DIR", "./smart_cache")

"""Backward-compatible re-exports — use session_manager.py for new code."""

from utils.session_manager import (  # noqa: F401
    SessionKeys,
    get_dataframe,
    get_dataset_name,
    get_metadata,
    has_dataset,
    init_session as init_session_state,
    reset_eda_pipeline as reset_eda_state,
    reset_model_pipeline as reset_model_state,
    reset_on_new_dataset,
    persist_dataset_metadata,
    set_dataframe,
    store_model_results,
)
from utils.pipeline_bridge import apply_autonomous_result_to_session

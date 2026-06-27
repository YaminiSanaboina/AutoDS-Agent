"""Cached expensive computations for EDA and health scoring."""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from utils.health_score import compute_health_score


def _df_fingerprint(df: pd.DataFrame) -> str:
    payload = f"{df.shape}|{list(df.columns)}|{int(df.isnull().sum().sum())}|{int(df.duplicated().sum())}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


@st.cache_data(show_spinner=False)
def cached_health_score(df_bytes: bytes, fingerprint: str, target_column: str | None) -> dict:
    """Return health score for a dataset fingerprint."""
    import io
    df = pd.read_parquet(io.BytesIO(df_bytes)) if df_bytes else pd.DataFrame()
    if df.empty:
        return compute_health_score(None)
    return compute_health_score(df, target_column=target_column or None)


def get_cached_health_score(df: pd.DataFrame, target_column: str | None = None) -> dict:
    """Cache wrapper that stores parquet bytes for Streamlit cache_data."""
    if df is None or df.empty:
        return compute_health_score(None)
    try:
        import io
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        target_key = target_column or ""
        return cached_health_score(buf.getvalue(), _df_fingerprint(df), target_key)
    except Exception:
        return compute_health_score(df, target_column=target_column)

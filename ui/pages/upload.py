import pandas as pd
import streamlit as st

from ui.components import dataset_loaded_banner
from utils.session_manager import get_dataframe, get_dataset_name, reset_on_new_dataset
from utils.styling import render_page_header


def render():
    render_page_header(
        "Upload Data",
        "Import your dataset to begin the AI analysis pipeline",
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx"],
        help="Supported formats: .csv, .xlsx",
    )

    if uploaded_file is None:
        st.markdown("---")
        st.subheader("Sample Datasets")
        st.caption("Quick-start with bundled sample data")

        samples = {
            "Housing.csv": "data/Housing.csv",
            "Titanic-Dataset.csv": "data/Titanic-Dataset.csv",
            "Telco Customer Churn.csv": "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        }

        cols = st.columns(3)
        for idx, (label, path) in enumerate(samples.items()):
            with cols[idx]:
                if st.button(f"Load {label}", key=f"sample_{idx}", width="stretch"):
                    try:
                        df = pd.read_csv(path)
                        _store_dataset(df, label)
                    except Exception as exc:
                        st.error(f"Unable to load sample dataset: {exc}")
        return

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        _store_dataset(df, uploaded_file.name)
    except Exception as exc:
        st.error(f"Unable to parse uploaded file: {exc}")


def _store_dataset(df, filename):
    reset_on_new_dataset(df, filename)
    st.success("Dataset loaded successfully!")
    dataset_loaded_banner(get_dataframe(), get_dataset_name())
    st.dataframe(get_dataframe().head(10), width="stretch", hide_index=True)

    loaded_df = get_dataframe()
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{loaded_df.shape[0]:,}")
    c2.metric("Columns", loaded_df.shape[1])
    c3.metric("Memory", f"{loaded_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

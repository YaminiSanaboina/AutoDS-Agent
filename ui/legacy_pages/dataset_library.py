import streamlit as st


from config import PRIMARY_COLOR
from utils.dataset_library import get_library_datasets
from utils.datasets import load_sample_dataset
from utils.session_manager import SessionKeys, reset_on_new_dataset
from utils.styles import render_hero


def render():
    render_hero(
        "Dataset Library",
        "Explore curated datasets, learn what each teaches, and start analysis in one click",
    )

    datasets = get_library_datasets()
    st.caption(


        f"**{len(datasets)} datasets** · Click **Open Dataset Source** for Kaggle/UCI links."
    )

    cols = st.columns(3)
    for idx, ds in enumerate(datasets):
        with cols[idx % 3]:
            _render_dataset_card(ds)


def _render_dataset_card(ds):
    card_html = f"""
    <div class="resource-card">
        <div style="font-size:2rem;margin-bottom:0.5rem">{ds['icon']}</div>
        <strong style="font-size:1.05rem">{ds['name']}</strong>
        <p style="color:#64748B;font-size:0.85rem;margin:0.5rem 0">{ds['description']}</p>

        <p style="color:#334155;font-weight:600;margin:0.75rem 0 0.25rem">
            Business Problem
        </p>
        <p style="color:#64748B;font-size:0.85rem;margin:0">{ds['business_problem']}</p>
        <p style="color:#334155;font-weight:600;margin:0.75rem 0 0.25rem">
            ML Tasks/Concepts
        </p>
        <ul style="color:#64748B;font-size:0.85rem;margin:0;padding-left:1.1rem">
            {''.join(f'<li>{task}</li>' for task in ds['ml_tasks'])}
        </ul>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.link_button(

            "🌐 Open Dataset Source",
            ds["source_url"],

            width="stretch"
        )
    with btn_col2:
        if st.button(
            "Load into AutoDS",

            key=f"lib_load_{ds['id']}",
            width="stretch",
            type="primary",
        ):

            df, name = load_sample_dataset(ds["id"])
            reset_on_new_dataset(df, name)
            st.session_state[SessionKeys.CURRENT_PAGE] = "data_hub"
            st.success(f"**{ds['name']}** loaded successfully! Opening Data Hub...")
            st.rerun()

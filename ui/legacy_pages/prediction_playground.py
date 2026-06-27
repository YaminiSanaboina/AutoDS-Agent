import pandas as pd
import streamlit as st

from ui.components import glass_panel, glass_panel_small, require_dataset
from utils.session_manager import SessionKeys, get_dataframe
from utils.styles import render_hero


def render():
    render_hero("Prediction Playground", "Test your trained model on new examples")

    if not require_dataset():
        return

    if not st.session_state.get(SessionKeys.MODEL_TRAINED):
        glass_panel_small("Train a model in **AutoML Studio** before running predictions.")
        return

    model = st.session_state.get(SessionKeys.BEST_MODEL)
    X = st.session_state.get(SessionKeys.X_DATA)
    best_name = st.session_state.get(SessionKeys.BEST_MODEL_NAME, "Model")
    problem_type = st.session_state.get(SessionKeys.PROBLEM_TYPE, "Classification")

    if model is None or X is None or getattr(X, "empty", True):
        st.warning("Model artifacts are missing. Re-train the model in AutoML Studio.")
        return

    glass_panel("Interactive Prediction", "Enter feature values and generate a model prediction.")

    feature_values = {}
    cols = st.columns(2)
    for idx, column in enumerate(X.columns):
        with cols[idx % 2]:
            series = X[column]
            if pd.api.types.is_numeric_dtype(series):
                default = float(series.median()) if series.notna().any() else 0.0
                feature_values[column] = st.number_input(str(column), value=default, key=f"pred_{column}")
            else:
                options = sorted(series.dropna().astype(str).unique().tolist())
                if not options:
                    options = [""]
                feature_values[column] = st.selectbox(str(column), options, key=f"pred_{column}")

    if st.button("Generate Prediction", type="primary"):
        try:
            input_df = pd.DataFrame([feature_values], columns=list(X.columns))
            prediction = model.predict(input_df)[0]
            st.session_state["prepared_input"] = feature_values
            st.session_state["last_prediction"] = prediction

            if hasattr(model, "predict_proba") and problem_type == "Classification":
                proba = model.predict_proba(input_df)[0]
                confidence = float(max(proba) * 100)
                st.session_state[SessionKeys.CONFIDENCE_SCORE] = round(confidence, 2)
                st.success(f"Prediction: **{prediction}** (confidence {confidence:.1f}%)")
            else:
                st.success(f"Prediction: **{prediction}**")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    last_prediction = st.session_state.get("last_prediction")
    if last_prediction is not None:
        st.markdown("---")
        st.subheader("Latest Result")
        st.metric("Model", best_name)
        st.metric("Prediction", str(last_prediction))
        confidence = st.session_state.get(SessionKeys.CONFIDENCE_SCORE)
        if confidence is not None:
            st.metric("Confidence", f"{confidence}%")

    df = get_dataframe()
    if df is not None and not df.empty:
        with st.expander("Preview a sample row from the dataset"):
            row_idx = st.number_input("Row index", min_value=0, max_value=max(len(df) - 1, 0), value=0, step=1)
            st.dataframe(df.iloc[[int(row_idx)]], width="stretch")

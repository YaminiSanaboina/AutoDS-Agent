"""Image Classification Demo - Standalone image analysis interface."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import streamlit as st
from PIL import Image

from utils.image_classifier import ImageClassifier, load_image_from_bytes, validate_image_format
from config import PRIMARY_COLOR, ACCENT_COLOR, SUCCESS_COLOR, WARNING_COLOR


def render_image_analysis_demo() -> None:
    """Render the Image Classification Demo interface."""
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="color: {0};">🖼️ Image Classification Demo</h1>
            <p style="font-size: 16px; color: #666;">
                Upload an image and let AI identify what's in it.
            </p>
        </div>
        """.format(PRIMARY_COLOR),
        unsafe_allow_html=True,
    )

    # Initialize session state
    if "image_classifier" not in st.session_state:
        st.session_state.image_classifier = None
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "mobilenetv2"

    # Model selection
    col1, col2 = st.columns([3, 1])
    with col1:
        model_choice = st.radio(
            "Select Model:",
            options=["mobilenetv2", "resnet50"],
            format_func=lambda x: "MobileNetV2 (Faster, Smaller)" if x == "mobilenetv2" else "ResNet50 (More Accurate)",
            horizontal=True,
            key="model_selector",
        )
    with col2:
        st.info("Model Info", icon="ℹ️")

    st.session_state.selected_model = model_choice

    # Image upload
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, PNG, GIF, BMP, WebP)",
        type=["jpg", "jpeg", "png", "gif", "bmp", "webp"],
        key="image_uploader",
    )

    if uploaded_file is not None:
        # Load and display image
        image_bytes = uploaded_file.read()
        image = load_image_from_bytes(image_bytes)

        if image is None:
            st.error("Failed to load image. Please try another file.")
            return

        st.session_state.uploaded_image = image

        # Display uploaded image
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.markdown("### 📸 Uploaded Image")
            st.image(image, use_column_width=True, caption=f"Size: {image.width}×{image.height}px")

        # Classification section
        with col2:
            st.markdown("### 🤖 Classification")

            if st.button("Classify Image", type="primary", width="stretch", key="classify_btn"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Initialize classifier if needed
                        if (
                            st.session_state.image_classifier is None
                            or st.session_state.image_classifier.model_name != model_choice
                        ):
                            st.session_state.image_classifier = ImageClassifier(model_choice)

                        classifier = st.session_state.image_classifier
                        predicted_class, confidence, details = classifier.predict(image)

                        st.session_state.last_prediction = {
                            "class": predicted_class,
                            "confidence": confidence,
                            "details": details,
                            "model": classifier.model_name_display,
                        }
                    except Exception as e:
                        st.error(f"Classification failed: {e}")
                        return

        # Display prediction results
        if st.session_state.last_prediction:
            _render_prediction_results(st.session_state.last_prediction)

        # Advanced options
        with st.expander("📊 Advanced Analysis", expanded=False):
            _render_advanced_analysis(image, st.session_state.last_prediction)

    else:
        # Empty state
        st.info(
            "👆 Upload an image to get started. The AI will identify objects, animals, scenes, and more!",
            icon="📝",
        )

        # Example information
        st.markdown("### 💡 How It Works")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
                **1. Upload**
                Upload any JPG or PNG image from your computer.
                """
            )
        with col2:
            st.markdown(
                """
                **2. Classify**
                AI analyzes the image using a pretrained neural network.
                """
            )
        with col3:
            st.markdown(
                """
                **3. Predict**
                See predicted class and confidence score instantly.
                """
            )


def _render_prediction_results(prediction: Dict[str, Any]) -> None:
    """Render the classification results."""
    st.markdown("### ✨ Results")

    col1, col2 = st.columns(2)

    with col1:
        # Predicted class with confidence bar
        predicted_class = prediction["class"]
        confidence = prediction["confidence"]

        st.markdown(f"**Predicted Class:** `{predicted_class}`")

        # Confidence gauge
        confidence_pct = min(100, max(0, confidence))
        if confidence_pct >= 80:
            color = SUCCESS_COLOR
            status = "High Confidence"
        elif confidence_pct >= 60:
            color = ACCENT_COLOR
            status = "Moderate Confidence"
        else:
            color = WARNING_COLOR
            status = "Low Confidence"

        st.markdown(
            f"""
            <div style="margin-top: 1rem;">
                <div style="font-size: 14px; margin-bottom: 0.5rem;">
                    Confidence Score: <strong>{confidence:.1f}%</strong> ({status})
                </div>
                <div style="
                    width: 100%;
                    height: 24px;
                    background-color: #e0e0e0;
                    border-radius: 12px;
                    overflow: hidden;
                    border: 2px solid {color};
                ">
                    <div style="
                        width: {confidence_pct}%;
                        height: 100%;
                        background: linear-gradient(90deg, {color}, {ACCENT_COLOR});
                        transition: width 0.3s ease;
                    "></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(f"**Model Used:** {prediction['model']}")
        st.markdown(f"**Input Size:** {prediction['details'].get('input_size', 'N/A')}")

    # Top 5 predictions
    if "top_5" in prediction["details"]:
        st.markdown("### 🏆 Top 5 Predictions")
        top_5 = prediction["details"]["top_5"]

        for i, pred in enumerate(top_5, 1):
            class_name = pred["class"]
            conf = pred["confidence"]
            st.progress(
                min(1.0, conf / 100),
                text=f"{i}. {class_name} ({conf:.1f}%)",
            )


def _render_advanced_analysis(image: Image.Image, prediction: Optional[Dict[str, Any]]) -> None:
    """Render advanced analysis options."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Image Properties:**")
        st.markdown(f"- **Size:** {image.width} × {image.height} pixels")
        st.markdown(f"- **Mode:** {image.mode}")
        st.markdown(f"- **Format:** {image.format or 'Unknown'}")

    with col2:
        if prediction:
            st.markdown("**Classification Details:**")
            model = prediction.get("model", "Unknown")
            st.markdown(f"- **Model:** {model}")
            st.markdown(f"- **Top Prediction:** {prediction['class']}")
            st.markdown(f"- **Confidence:** {prediction['confidence']:.1f}%")

    # Color histogram
    st.markdown("**Color Distribution:**")
    col1, col2, col3 = st.columns(3)

    img_array = image.convert("RGB")

    with col1:
        st.metric("🔴 Red", f"{_get_channel_mean(img_array, 0):.0f}", "")

    with col2:
        st.metric("🟢 Green", f"{_get_channel_mean(img_array, 1):.0f}", "")

    with col3:
        st.metric("🔵 Blue", f"{_get_channel_mean(img_array, 2):.0f}", "")


def _get_channel_mean(image: Image.Image, channel: int) -> float:
    """Get mean value for a color channel."""
    try:
        import numpy as np

        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] > channel:
            return np.mean(img_array[:, :, channel])
        return 0.0
    except Exception:
        return 0.0

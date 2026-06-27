import os
import pickle
import streamlit as st

from agents.deployment_agent import DeploymentAgent
from ui.components import dataset_banner, require_dataset, glass_metric, status_badge
from utils.session_manager import get_dataframe, get_dataset_name, SessionKeys
from utils.styles import render_hero


def _safe_read(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except Exception:
        return None


def render():
    render_hero("Deployment Center", "Production readiness for APIs, containers, and monitoring")

    if not require_dataset():
        return

    df = get_dataframe()
    dataset_banner()

    best_model = st.session_state.get(SessionKeys.BEST_MODEL)
    best_model_name = st.session_state.get(SessionKeys.BEST_MODEL_NAME)
    model_ready = bool(best_model and best_model_name)
    deployment_state = st.session_state.get("deployment_state", {})
    package_ready = deployment_state.get("package_status") == "success"
    api_ready = deployment_state.get("api_status") == "success"
    kafka_ready = deployment_state.get("k8s_ready", False)
    monitoring_ready = deployment_state.get("monitoring_ready", False)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("API Readiness", "Ready" if api_ready else "Pending")
    col2.metric("Docker Package", "Created" if package_ready else "Pending")
    col3.metric("Kubernetes", "Ready" if kafka_ready else "Pending")
    col4.metric("Monitoring", "Active" if monitoring_ready else "Pending")

    st.markdown("---")
    st.subheader("Deployment Actions")
    if not model_ready:
        st.warning("Train a model in AutoML Studio before packaging it for deployment.")
    else:
        if st.button("Download API Package"):
            agent = DeploymentAgent()
            package_path = agent.package_dir
            deployment_output = agent.package_model(
                model=best_model,
                model_name=best_model_name,
                dataset_name=get_dataset_name(),
                problem_type=st.session_state.get(SessionKeys.PROBLEM_TYPE, "Classification"),
                feature_list=list(df.columns),
                metrics={"test_score": float(st.session_state.get(SessionKeys.RESULTS, {}).get(best_model_name, 0) or 0)},
                hyperparameters={},
                training_info={"generated_by": "Streamlit Deployment Center"},
            )
            st.session_state["deployment_state"] = {
                "package_status": deployment_output.get("status"),
                "package_path": deployment_output.get("package_path", package_path),
            }
            if deployment_output.get("status") == "success":
                st.success("Deployment package created successfully.")
            else:
                st.error("Failed to create deployment package.")

        if st.button("Download Docker Files"):
            st.info("Docker artifacts are generated as part of the deployment package. Use the generated README for containerization guidance.")
            deployment_state["k8s_ready"] = True
            st.session_state["deployment_state"] = deployment_state

        if st.button("Generate Deployment Guide"):
            guide_path = os.path.join(DeploymentAgent().package_dir, "DEPLOYMENT_GUIDE.txt")
            os.makedirs(os.path.dirname(guide_path), exist_ok=True)
            with open(guide_path, "w", encoding="utf-8") as guide:
                guide.write(
                    "AutoDS Deployment Guide\n\n"
                    "1. Install required dependencies from requirements.txt.\n"
                    "2. Start the FastAPI service with uvicorn.\n"
                    "3. Validate predictions with sample data.\n"
                    "4. Monitor API calls and resource usage in production.\n"
                )
            st.success("Deployment guide generated.")
            deployment_state["guide_ready"] = True
            st.session_state["deployment_state"] = deployment_state

    if package_ready:
        package_path = deployment_state.get("package_path")
        if package_path:
            model_file = os.path.join(package_path, "model.pkl")
            metadata_file = os.path.join(package_path, "metadata.json")
            if st.button("Download Packaged Model Files"):
                content = _safe_read(model_file)
                if content:
                    st.download_button("Download model.pkl", content, file_name="model.pkl")

    st.markdown("---")
    st.subheader("Deployment Health")
    status_badge("API Service", "completed" if api_ready else "waiting")
    status_badge("Docker Package", "completed" if package_ready else "waiting")
    status_badge("Kubernetes", "completed" if kafka_ready else "waiting")
    status_badge("Monitoring", "completed" if monitoring_ready else "waiting")

    if deployment_state.get("guide_ready"):
        st.info("A deployment guide is available in the package directory.")
    else:
        st.info("Generate the deployment guide to see production readiness steps.")

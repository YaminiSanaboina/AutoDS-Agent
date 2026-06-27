import pandas as pd
import streamlit as st

from agents.dataset_agent import analyze_dataset, format_ai_explanation
from ui.components import ai_assistant_panel, ai_chat_message, dataset_banner, require_dataset
from utils.datasets import get_sample_datasets, load_sample_dataset
from utils.health_score import compute_health_score
from utils.session_manager import (
    SessionKeys,
    get_dataframe,
    get_metadata,
    reset_on_new_dataset,
)
from utils.styles import render_hero


def render():
    render_hero("Data Hub", "Upload, preview, and manage your datasets")

    tab_upload, tab_workspace = st.tabs(
        ["Upload & Resources", "Dataset Workspace"]
    )

    with tab_upload:
        _render_upload_section()
        st.markdown("---")
        _render_resources()

    with tab_workspace:
        if not require_dataset():
            return
        _render_workspace()


def _render_upload_section():
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"],
        key="file_uploader_main",
    )

    if uploaded_file is not None:
        upload_key = f"{uploaded_file.name}_{uploaded_file.size}"

        if st.session_state.get("last_upload_key") != upload_key:
            df = (
                pd.read_csv(uploaded_file)
                if uploaded_file.name.endswith(".csv")
                else pd.read_excel(uploaded_file)
            )

            reset_on_new_dataset(df, uploaded_file.name)
            st.session_state["last_upload_key"] = upload_key

            st.success("Dataset uploaded and saved to session!")

    df = get_dataframe()

    if df is not None:
        dataset_banner()
        _render_ai_dataset_understanding(df)
        st.dataframe(df.head(10), width="stretch", hide_index=True)
    else:
        st.info("Upload a CSV/Excel file or load a sample dataset below.")

    _render_dataset_sources()


def _render_dataset_sources():
    st.markdown("---")
    with st.expander("🌍 Dataset Resources"):
        st.write("Online dataset links for AI and machine learning projects.")

        resources = [
            ("Kaggle Datasets", "https://www.kaggle.com/datasets"),
            ("UCI Machine Learning Repository", "https://archive.ics.uci.edu/ml"),
            ("Google Dataset Search", "https://datasetsearch.research.google.com"),
            ("Government Open Data", "https://data.gov/"),
            ("GitHub Public Datasets", "https://github.com/awesomedata/awesome-public-datasets"),
            ("AWS Open Data Registry", "https://registry.opendata.aws"),
            ("Hugging Face Datasets", "https://huggingface.co/datasets"),
        ]

        cols = st.columns(2)

        for i, (name, url) in enumerate(resources):
            with cols[i % 2]:
                st.link_button(
                    f"🌐 {name}",
                    url,
                    width="stretch",
                )


def _render_resources():
    st.markdown("### 🌍 Dataset Resources")
    st.caption("One-click sample datasets for testing the full pipeline")

    datasets = get_sample_datasets()
    
    # Enrich dataset metadata with real-world problems and ML concepts
    dataset_details = {
        "titanic": {
            "problem": "Predict passenger survival patterns",
            "concepts": ["Classification", "Binary Classification", "Feature Engineering"]
        },
        "churn": {
            "problem": "Identify customers likely to leave a service",
            "concepts": ["Classification", "Imbalanced Data", "Feature Engineering"]
        },
        "housing": {
            "problem": "Estimate property prices based on characteristics",
            "concepts": ["Regression", "Numerical Prediction", "Feature Scaling"]
        },
        "iris": {
            "problem": "Classify flowers by species from measurements",
            "concepts": ["Multi-class Classification", "Feature Engineering", "Data Scaling"]
        },
        "wine": {
            "problem": "Classify wine quality from chemical properties",
            "concepts": ["Classification", "Regression", "Feature Engineering"]
        },
        "heart": {
            "problem": "Diagnose heart disease from medical indicators",
            "concepts": ["Binary Classification", "Medical Prediction", "Feature Engineering"]
        }
    }
    
    cols = st.columns(3)

    for idx, ds in enumerate(datasets):
        with cols[idx % 3]:
            details = dataset_details.get(ds['id'], {
                "problem": "Analyze this dataset to discover patterns",
                "concepts": ["Data Analysis", "Machine Learning"]
            })
            
            concepts_str = " • ".join(details["concepts"])
            
            st.markdown(
                f"""
                <div class="resource-card">
                    <div style="font-size:2rem;">{ds['icon']}</div>
                    <strong>{ds['name']}</strong>
                    <p style="color:#475569;font-size:0.9rem;margin:0.5rem 0;">
                        <strong>Real-world Problem:</strong><br>
                        {details['problem']}
                    </p>
                    <p style="color:#6366F1;font-size:0.8rem;margin:0.5rem 0;">
                        <strong>ML Concepts:</strong><br>
                        {concepts_str}
                    </p>
                    <p style="color:#64748B;font-size:0.85rem;">
                        {ds['description']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                f"Load {ds['name']}",
                key=f"load_{ds['id']}",
                width="stretch",
            ):
                df, name = load_sample_dataset(ds["id"])
                reset_on_new_dataset(df, name)
                st.rerun()

            with open(ds["path"], "rb") as f:
                st.download_button(
                    "Download CSV",
                    f.read(),
                    file_name=ds["filename"],
                    key=f"dl_{ds['id']}",
                    width="stretch",
                )


def _get_dataset_analysis(df):
    meta = get_metadata() or {}
    analysis = meta.get("analysis")

    if analysis and "dataset_overview" in analysis:
        return analysis

    try:
        analysis = analyze_dataset(df)
        health = compute_health_score(df)
        st.session_state[SessionKeys.DATASET_METADATA] = {
            **meta,
            "analysis": analysis,
            "health": health,
            "health_score": health.get("score"),
        }
        return analysis
    except Exception as exc:
        st.warning(f"Dataset analysis unavailable: {exc}")
        return {
            "dataset_overview": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "memory_mb": 0,
                "column_names": list(df.columns),
            },
            "column_information": {},
            "data_quality_issues": {"missing_values": {"total": 0}, "duplicate_rows": 0},
            "ml_understanding": {},
            "ai_recommendations": ["Review the dataset preview and continue to Data Quality Lab."],
            "next_step": "Go to Data Quality Lab.",
        }


def _infer_dataset_context(column_names):
    """Infer what a dataset represents based on column names."""
    col_str = " ".join(str(c).lower() for c in column_names)
    
    contexts = {
        "titanic": (["passenger", "survived", "embarked", "sibsp"], "passenger survival patterns"),
        "churn": (["churn", "tenure", "contract", "monthly"], "customer retention and service usage"),
        "housing": (["price", "square", "bedrooms", "garage"], "property characteristics and pricing"),
        "medical": (["age", "blood", "heart", "pressure", "diagnosis", "disease"], "health indicators and medical diagnosis"),
        "customer": (["customer", "purchased", "spent", "subscription"], "customer behavior and purchasing patterns"),
        "finance": (["revenue", "profit", "sales", "cost", "income"], "financial metrics and business performance"),
        "classification": (["survived", "churn", "default", "class", "category", "status", "label"], "categorical prediction targets"),
        "regression": (["price", "sales", "amount", "revenue", "cost", "score"], "numerical value prediction"),
    }
    
    detected_contexts = []
    for context_type, (keywords, description) in contexts.items():
        if any(kw in col_str for kw in keywords):
            detected_contexts.append((context_type, description))
    
    return detected_contexts


def _generate_intelligent_summary(df, columns_info, rows, cols, memory):
    """Generate an intelligent, context-aware dataset summary."""
    column_names = list(df.columns)
    contexts = _infer_dataset_context(column_names)
    
    # Build a smart summary
    if contexts:
        # Get the most likely context
        context_type, description = contexts[0]
        summary = f"This dataset contains information about {rows:,} observations with {cols} features related to {description}."
    else:
        summary = f"This dataset has {rows:,} rows and {cols} columns with diverse data types and potential patterns."
    
    return summary


def _detect_problem_type_from_columns(column_names, suggested_target=None):
    """Intelligently detect problem type from column names and target."""
    col_names_lower = [str(c).lower() for c in column_names]
    
    # Classification keywords
    classification_keywords = ["survived", "churn", "default", "diagnosis", "outcome", "class", 
                             "category", "label", "status", "approved", "risk", "type", "fraud", "disease"]
    
    # Regression keywords
    regression_keywords = ["price", "sales", "revenue", "amount", "cost", "value", "salary", 
                          "income", "score", "rating", "rate", "age", "weight", "height"]
    
    # Check target column first
    if suggested_target:
        target_lower = str(suggested_target).lower()
        if any(keyword in target_lower for keyword in classification_keywords):
            return "classification", "target"
        if any(keyword in target_lower for keyword in regression_keywords):
            return "regression", "target"
    
    # Check all column names
    has_classification = any(keyword in col for col in col_names_lower for keyword in classification_keywords)
    has_regression = any(keyword in col for col in col_names_lower for keyword in regression_keywords)
    
    if has_classification and not has_regression:
        return "classification", "columns"
    elif has_regression and not has_classification:
        return "regression", "columns"
    elif has_classification and has_regression:
        # Mixed signals - prefer classification if more classification keywords
        class_count = sum(1 for col in col_names_lower for keyword in classification_keywords if keyword in col)
        reg_count = sum(1 for col in col_names_lower for keyword in regression_keywords if keyword in col)
        if class_count > reg_count:
            return "classification", "columns"
        else:
            return "regression", "columns"
    
    return "unknown", "none"


def _build_ml_explanation(column_names, rows, suggested_target, problem_type, possible_targets):
    """Build an intelligent ML explanation with detected problem type."""
    inferred_type, detection_source = _detect_problem_type_from_columns(column_names, suggested_target)
    
    # Override problem type if we can infer from columns
    if inferred_type != "unknown" and problem_type == "unknown":
        problem_type = inferred_type
    
    ml_detail = "Machine Learning Potential: "
    
    if problem_type == "classification":
        ml_detail += "This dataset is suitable for Classification tasks. "
        if suggested_target:
            ml_detail += f"The likely target column is '{suggested_target}', which represents categories or classes to predict. "
            ml_detail += "The model will learn patterns to classify new observations into these categories."
        elif possible_targets:
            target_list = ", ".join(f"'{str(t)}'" for t in possible_targets[:2])
            ml_detail += f"Possible target columns include {target_list}. "
            ml_detail += "Select one in AutoML Studio to build classification models."
        else:
            ml_detail += "Select a target column in AutoML Studio to train classification models that predict categories."
    
    elif problem_type == "regression":
        ml_detail += "This dataset is suitable for Regression tasks. "
        if suggested_target:
            ml_detail += f"The likely target column is '{suggested_target}', which represents a continuous numerical value to predict. "
            ml_detail += "The model will learn patterns to estimate this value for new observations."
        elif possible_targets:
            target_list = ", ".join(f"'{str(t)}'" for t in possible_targets[:2])
            ml_detail += f"Possible target columns include {target_list}. "
            ml_detail += "Select one in AutoML Studio to build regression models."
        else:
            ml_detail += "Select a target column in AutoML Studio to train regression models that predict numerical values."
    
    else:
        ml_detail += "Explore this dataset using EDA Explorer to understand its structure and patterns. "
        ml_detail += "After selecting a target column in AutoML Studio, the system can recommend the best machine learning approach."
    
    return ml_detail


def _render_ai_dataset_understanding(df):
    analysis = _get_dataset_analysis(df)
    
    # Extract analysis components
    overview = analysis.get("dataset_overview", {})
    columns = analysis.get("column_information", {})
    quality = analysis.get("data_quality_issues", {})
    ml = analysis.get("ml_understanding", {})
    
    rows = overview.get("rows", 0)
    cols = overview.get("columns", 0)
    memory = overview.get("memory_mb", 0)
    column_names = overview.get("column_names", [])
    
    # TASK 1: Generate intelligent, context-aware summary
    summary = _generate_intelligent_summary(df, columns, rows, cols, memory)
    
    # Build details list
    details = []

    col_info = analysis.get("column_information", {})
    numerical = col_info.get("numerical") or []
    categorical = col_info.get("categorical") or []
    if numerical:
        preview = ", ".join(str(c) for c in numerical[:5])
        details.append(f"Numerical columns ({len(numerical)}): {preview}")
    if categorical:
        preview = ", ".join(str(c) for c in categorical[:5])
        details.append(f"Categorical columns ({len(categorical)}): {preview}")

    suggested_target = ml.get("suggested_target")
    possible_targets = ml.get("possible_targets") or []
    problem_type = ml.get("problem_type", "unknown")
    details.append(
        _build_ml_explanation(column_names, rows, suggested_target, problem_type, possible_targets)
    )

    missing_total = quality.get("missing_values", {}).get("total", 0)
    duplicates = quality.get("duplicate_rows", 0)
    if missing_total:
        details.append(f"Missing values detected: {missing_total:,} cells.")
    if duplicates:
        details.append(f"Duplicate rows detected: {duplicates:,}.")

    recommendations = analysis.get("ai_recommendations", [])
    recommendation = recommendations[0] if recommendations else analysis.get("next_step")

    ai_assistant_panel(
        summary,
        details,
        recommendation=recommendation,
        title="AI Dataset Understanding",
    )

    try:
        explanation_html = format_ai_explanation(analysis)
        if explanation_html:
            st.markdown("### Detailed AI Explanation")
            st.markdown(explanation_html, unsafe_allow_html=True)
    except Exception as exc:
        st.caption(f"Detailed explanation unavailable: {exc}")


def _render_workspace():
    df = get_dataframe()
    if df is None:
        st.info("Upload a dataset in the Upload tab to open the workspace.")
        return

    try:
        health = compute_health_score(df)
        analysis = _get_dataset_analysis(df)
    except Exception as exc:
        st.warning(f"Workspace profiling unavailable: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{analysis.get('rows', df.shape[0]):,}")
    c2.metric("Columns", analysis.get("columns", df.shape[1]))
    c3.metric("Duplicates", analysis.get("duplicate_rows", int(df.duplicated().sum())))
    c4.metric("Health", f"{health['score']}/100")

    st.caption(health.get("summary", ""))

    tab_overview, tab_types, tab_missing, tab_stats = st.tabs(
        ["Overview", "Data Types", "Missing Values", "Statistics"]
    )

    with tab_overview:
        st.write("**Column Names:**")
        column_names = analysis.get("column_names") or list(df.columns)
        st.code(", ".join(str(c) for c in column_names), language=None)
        st.dataframe(df.head(15), width="stretch", hide_index=True)

    with tab_types:
        dtypes = analysis.get("data_types")
        if dtypes is not None and hasattr(dtypes, "reset_index"):
            types_df = dtypes.reset_index()
            types_df.columns = ["Column", "Data Type"]
            st.dataframe(types_df, width="stretch", hide_index=True)
        else:
            st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Data Type"}), width="stretch")

    with tab_missing:
        missing_values = analysis.get("missing_values")
        if missing_values is not None and hasattr(missing_values, "reset_index"):
            missing_df = missing_values.reset_index()
            missing_df.columns = ["Column", "Missing Count"]
            missing_df["Missing %"] = (missing_df["Missing Count"] / max(len(df), 1) * 100).round(2)
            st.dataframe(missing_df, width="stretch", hide_index=True)
        else:
            st.info("Missing value breakdown is not available for this dataset.")

    with tab_stats:
        summary = analysis.get("summary")
        if summary is not None:
            st.dataframe(summary, width="stretch")
        else:
            st.dataframe(df.describe(include="all"), width="stretch")
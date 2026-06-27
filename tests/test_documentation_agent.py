import json
import os

from agents.documentation_agent import DocumentationAgent


def test_generate_project_documentation():
    agent = DocumentationAgent(knowledge_path="tests/knowledge_base_test.json")
    project_data = {
        "project_title": "Sales Forecasting",
        "business_objective": "Predict sales to improve inventory planning.",
        "dataset_name": "Sales Data",
        "dataset_summary": "This dataset captures daily sales, promotions, and product metadata.",
        "preprocessing_steps": "Missing values were imputed and categorical features encoded.",
        "feature_engineering_decisions": "Created lag features and rolling averages.",
        "model_selection_rationale": "Random Forest was selected for robustness and interpretability.",
        "hyperparameter_summary": "Grid search tuned depth and number of estimators.",
        "evaluation_metrics": "RMSE and MAE were used for regression performance.",
        "explainability_summary": "Feature importance and SHAP values explain predictions.",
        "deployment_recommendation": "Deploy as a batch scoring service to begin.",
        "future_improvements": "Add seasonal trend features and monitor drift.",
    }
    document = agent.generate_project_documentation(project_data)
    assert document["title"] == "Sales Forecasting"
    assert "Business Objective" in document["sections"]
    assert document["summary"]


def test_generate_dataset_card():
    agent = DocumentationAgent(knowledge_path="tests/knowledge_base_test.json")
    dataset_report = {
        "dataset_name": "Customer Churn",
        "source": "Internal CRM",
        "samples": 1200,
        "features": ["age", "tenure", "subscription_type"],
        "target_variable": "churn",
        "domain": "Telecom",
        "quality_score": 88,
        "risks": ["Imbalanced target."],
        "recommended_usage": "Use for churn prediction and retention analysis.",
    }
    card = agent.generate_dataset_card(dataset_report)
    assert card["dataset_name"] == "Customer Churn"
    assert card["quality_score"] == 88


def test_generate_model_card_with_ethics():
    agent = DocumentationAgent(knowledge_path="tests/knowledge_base_test.json")
    model_info = {
        "model_name": "ChurnClassifier",
        "framework": "sklearn",
        "version": "1.0",
        "description": "Binary classifier for churn.",
        "training_configuration": {"algorithm": "RandomForest", "trees": 100},
        "strengths": ["Good precision on the positive class."],
        "limitations": ["Needs periodic retraining."],
        "deployment_readiness": "Ready for staging deployment.",
    }
    metrics = {"accuracy": 0.92, "f1_score": 0.88}
    ethics_report = {
        "fairness_considerations": ["Monitor for demographic parity."],
        "responsible_ai_notes": ["Avoid using sensitive attributes."]
    }
    card = agent.generate_model_card(model_info, metrics, ethics_report=ethics_report)
    assert card["model_details"]["name"] == "ChurnClassifier"
    assert card["metrics"]["accuracy"] == 0.92
    assert card["fairness_considerations"] == ethics_report["fairness_considerations"]


def test_knowledge_base_save_and_search():
    path = "tests/knowledge_base_test.json"
    if os.path.exists(path):
        os.remove(path)

    agent = DocumentationAgent(knowledge_path=path)
    entry = agent.save_knowledge_entry(
        category="Research",
        title="Feature selection best practices",
        content="Use correlation and importance metrics to pick features.",
        tags=["features", "selection"],
    )

    assert entry["title"] == "Feature selection best practices"
    results = agent.search_knowledge("correlation")
    assert len(results) >= 1
    assert results[0]["title"] == "Feature selection best practices"

    entries = agent.list_entries(limit=5)
    assert len(entries) >= 1


def test_export_functions():
    agent = DocumentationAgent(knowledge_path="tests/knowledge_base_test.json")
    document = {
        "title": "Test Report",
        "summary": "A brief summary.",
        "sections": {
            "Overview": "This is an overview.",
            "Details": {"part_a": "A", "part_b": "B"},
        },
    }
    markdown = agent.export_markdown(document)
    assert "# Test Report" in markdown
    assert "## Overview" in markdown

    json_text = agent.export_json(document)
    parsed = json.loads(json_text)
    assert parsed["title"] == "Test Report"

    plain_text = agent.export_text(document)
    assert "Overview:" in plain_text
    assert "Details:" in plain_text

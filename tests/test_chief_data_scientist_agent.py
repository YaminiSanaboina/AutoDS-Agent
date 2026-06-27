import os
import tempfile

import numpy as np
import pandas as pd

from agents.chief_data_scientist_agent import ChiefDataScientistAgent


def test_analyze_project_generates_summary_and_persists_decisions(tmp_path):
    dataset = pd.DataFrame(
        {
            "feature_a": np.random.rand(50),
            "feature_b": np.random.randint(0, 3, size=50),
            "target": np.random.choice([0, 1], size=50),
        }
    )
    metadata = {"name": "Test Dataset", "missing_percent": 0.0, "duplicate_percent": 0.0}
    decision_file = tmp_path / "decisions.json"

    chief = ChiefDataScientistAgent(decision_path=str(decision_file))
    summary = chief.analyze_project(
        dataset=dataset,
        dataset_metadata=metadata,
        model_history=[{"test_score": 0.8, "train_score": 0.82}],
        drift_history=[{"severity": "Low"}],
        deployment_status={"deployed": False, "shap_explained": False},
    )

    assert summary["project_stage"] in {"Model Validation", "Model Development", "Exploratory Analysis"}
    assert summary["project_health"] >= 0
    assert summary["next_best_action"]
    assert decision_file.exists()
    assert len(chief.get_decision_history()) >= 1


def test_delegate_project_tasks_returns_ordered_agents_based_on_risks(tmp_path):
    chief = ChiefDataScientistAgent(decision_path=str(tmp_path / "decisions.json"))
    summary = {
        "project_stage": "Model Optimization",
        "next_best_action": "Optimize hyperparameters and address overfitting.",
        "risks": ["High missing data (15.0%). Review and address."],
    }

    plan = chief.delegate_project_tasks(project_summary=summary)

    assert "HyperparameterOptimizationAgent" in plan["ordered_agents"]
    assert plan["project_stage"] == "Model Optimization"
    assert plan["ordered_agents"]


def test_generate_executive_report_contains_recommendations_and_confidence(tmp_path):
    chief = ChiefDataScientistAgent(decision_path=str(tmp_path / "decisions.json"))
    project_summary = {
        "project_stage": "Production",
        "project_health": 88.1,
        "health_label": "Production Ready",
        "strengths": ["Strong model performance."],
        "weaknesses": ["No deployment."],
        "risks": ["Low drift."],
        "next_best_action": "Deploy the current best model to production.",
    }
    recommendations = [
        {"recommendation": "Deploy to production.", "priority": "Medium"}
    ]
    delegation_plan = {"project_stage": "Production", "ordered_agents": ["DeploymentAgent"], "reasoning": "Deploy model."}

    report = chief.generate_executive_report(
        project_summary=project_summary,
        strategic_advice=recommendations,
        delegation_plan=delegation_plan,
    )

    assert report["title"] == "Chief Data Scientist Executive Report"
    assert report["confidence"] == "High"
    assert report["execution_plan"] == delegation_plan


def test_simulate_autonomous_cycle_includes_expected_steps(tmp_path):
    dataset = pd.DataFrame(
        {
            "feature_a": np.random.rand(20),
            "feature_b": np.random.randint(0, 5, size=20),
            "target": np.random.choice([0, 1], size=20),
        }
    )
    chief = ChiefDataScientistAgent(decision_path=str(tmp_path / "decisions.json"))
    steps = chief.simulate_autonomous_cycle(dataset=dataset, dataset_metadata={"name": "Simulation Dataset"})

    expected_steps = {
        "Dataset arrival",
        "Analyze dataset",
        "Generate feature engineering plan",
        "Train models",
        "Optimize hyperparameters",
        "Explain model",
        "Prepare deployment",
        "Monitor drift",
        "Retrain when needed",
    }

    assert {step["step"] for step in steps} >= expected_steps

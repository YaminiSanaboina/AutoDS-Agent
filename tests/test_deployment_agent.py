import os
import sys
import tempfile

sys.path.insert(0, os.getcwd())

import json
import pickle
from datetime import datetime
from sklearn.linear_model import LinearRegression
import pandas as pd

from agents.deployment_agent import DeploymentAgent


def test_package_model():
    with tempfile.TemporaryDirectory() as tempdir:
        model = LinearRegression()
        X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        y = pd.Series([7, 8, 9])
        model.fit(X, y)

        agent = DeploymentAgent(package_dir=tempdir, registry_path=os.path.join(tempdir, "registry.json"))
        result = agent.package_model(
            model=model,
            model_name="Linear Regression",
            dataset_name="Test Dataset",
            problem_type="Regression",
            feature_list=["feature1", "feature2"],
            metrics={"test_score": 0.95, "train_score": 0.97},
            hyperparameters={"fit_intercept": True},
            training_info={"training_time": 0.1, "preprocessing_steps": ["none"], "explainability_available": True},
            intended_usage="Predict numerical target values.",
            limitations="Do not use with unseen categorical features.",
            ethical_considerations="Monitor for drift and fairness.",
        )

        assert result["status"] == "success"
        files = result["files"]
        assert os.path.exists(files["model"])
        assert os.path.exists(files["metadata"])
        assert os.path.exists(files["requirements"])
        assert os.path.exists(files["readme"])

        with open(files["metadata"], "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        assert metadata["model_name"] == "Linear Regression"
        assert metadata["dataset_name"] == "Test Dataset"
        assert metadata["problem_type"] == "Regression"
        assert metadata["feature_list"] == ["feature1", "feature2"]
        assert metadata["version"].startswith("MODEL_v")


def test_generate_api_and_docker_files():
    with tempfile.TemporaryDirectory() as tempdir:
        model = LinearRegression()
        X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        y = pd.Series([7, 8, 9])
        model.fit(X, y)

        agent = DeploymentAgent(package_dir=tempdir, registry_path=os.path.join(tempdir, "registry.json"))
        package_result = agent.package_model(
            model=model,
            model_name="Linear Regression",
            dataset_name="Test Dataset",
            problem_type="Regression",
            feature_list=["feature1", "feature2"],
            metrics={"test_score": 0.95},
            hyperparameters={"fit_intercept": True},
            training_info={"training_time": 0.1, "preprocessing_steps": ["none"], "explainability_available": False},
        )
        assert package_result["status"] == "success"

        api_result = agent.generate_api(package_path=tempdir)
        assert api_result["status"] == "success"
        assert os.path.exists(api_result["app_file"])
        assert os.path.exists(api_result["requirements_file"])

        docker_result = agent.generate_docker_files(package_path=tempdir)
        assert docker_result["status"] == "success"
        assert os.path.exists(docker_result["dockerfile"])
        assert os.path.exists(docker_result["docker_compose"])


def test_generate_model_card_and_version_registry():
    with tempfile.TemporaryDirectory() as tempdir:
        agent = DeploymentAgent(package_dir=tempdir, registry_path=os.path.join(tempdir, "registry.json"))
        card = agent.generate_model_card(
            model_name="Linear Regression",
            dataset_name="Test Dataset",
            problem_type="Regression",
            training_date="2026-06-13T00:00:00Z",
            metrics={"test_score": 0.95, "train_score": 0.97},
            features=["feature1", "feature2"],
            version="MODEL_v1.0",
        )
        assert "# Model Card: Linear Regression" in card
        assert "## Performance Metrics" in card
        assert "feature1" in card

        version_entry = agent.create_version(
            model_name="Linear Regression",
            changes="Refined training workflow.",
            performance_improvement=0.06,
            package_path=tempdir,
        )
        assert version_entry["version"].startswith("MODEL_v")
        registry_file = os.path.join(tempdir, "registry.json")
        assert os.path.exists(registry_file)

        with open(registry_file, "r", encoding="utf-8") as handle:
            registry = json.load(handle)
        assert "LINEAR_REGRESSION" in registry


def test_analyze_deployment_risk():
    agent = DeploymentAgent(package_dir=tempfile.mkdtemp(), registry_path=os.path.join(tempfile.mkdtemp(), "registry.json"))
    metadata = {
        "metrics": {"test_score": 0.65, "train_score": 0.92},
        "training_info": {},
        "feature_list": [],
        "hyperparameters": {},
    }
    risk = agent.analyze_deployment_risk(metadata)
    assert risk["risk_level"] in {"Low", "Medium", "High"}
    assert len(risk["warnings"]) >= 1
    assert len(risk["recommendations"]) >= 1


if __name__ == "__main__":
    test_package_model()
    test_generate_api_and_docker_files()
    test_generate_model_card_and_version_registry()
    test_analyze_deployment_risk()
    print("ALL deployment agent tests passed")

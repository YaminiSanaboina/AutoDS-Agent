import json
import os
import pickle
import tempfile

from sklearn.linear_model import LogisticRegression

from agents.model_registry_agent import ModelRegistryAgent


def test_register_model_and_persist_registry():
    with tempfile.TemporaryDirectory() as tempdir:
        registry_path = os.path.join(tempdir, "model_registry.json")
        agent = ModelRegistryAgent(registry_path=registry_path)

        model = LogisticRegression()
        entry = agent.register_model(
            model=model,
            model_name="TestModel",
            dataset_name="Heart Disease",
            problem_type="Classification",
            algorithm="Logistic Regression",
            metrics={"accuracy": 0.92, "precision": 0.90, "recall": 0.88, "f1": 0.89, "roc_auc": 0.93},
            feature_names=["age", "cholesterol"],
            hyperparameters={"solver": "lbfgs"},
            training_time=12.4,
            shap_available=True,
            deployment_status="Not Deployed",
            artifact_path="/tmp/model.pkl",
        )

        assert entry["model_id"] == "MODEL_001"
        assert entry["version"] == "MODEL_v1"
        assert entry["model_name"] == "TestModel"
        assert entry["shap_available"] is True

        with open(registry_path, "r", encoding="utf-8") as handle:
            contents = json.load(handle)
        assert contents["models"][0]["model_name"] == "TestModel"


def test_get_model_versions_and_compare_versions():
    with tempfile.TemporaryDirectory() as tempdir:
        registry_path = os.path.join(tempdir, "model_registry.json")
        agent = ModelRegistryAgent(registry_path=registry_path)

        model = LogisticRegression()
        entry1 = agent.register_model(
            model=model,
            model_name="TestModel",
            dataset_name="Heart Disease",
            problem_type="Classification",
            algorithm="Logistic Regression",
            metrics={"accuracy": 0.92, "precision": 0.90, "recall": 0.88, "f1": 0.89, "roc_auc": 0.93},
            feature_names=["age", "cholesterol"],
            hyperparameters={"solver": "lbfgs"},
            training_time=12.4,
        )
        entry2 = agent.register_model(
            model=model,
            model_name="TestModel",
            dataset_name="Heart Disease",
            problem_type="Classification",
            algorithm="XGBoost",
            metrics={"accuracy": 0.96, "precision": 0.95, "recall": 0.94, "f1": 0.95, "roc_auc": 0.97},
            feature_names=["age", "cholesterol"],
            hyperparameters={"learning_rate": 0.1},
            training_time=18.2,
        )

        versions = agent.get_model_versions("TestModel")
        assert len(versions) == 2
        assert versions[0]["version"] == "MODEL_v1"
        assert versions[1]["version"] == "MODEL_v2"

        comparison = agent.compare_versions(entry1["model_id"], entry2["model_id"])
        assert "recommendation" in comparison
        assert "Upgrade to" in comparison["recommendation"]


def test_rollback_model():
    with tempfile.TemporaryDirectory() as tempdir:
        registry_path = os.path.join(tempdir, "model_registry.json")
        agent = ModelRegistryAgent(registry_path=registry_path)

        model = LogisticRegression()
        entry1 = agent.register_model(
            model=model,
            model_name="TestModel",
            dataset_name="Heart Disease",
            problem_type="Classification",
            algorithm="Logistic Regression",
            metrics={"accuracy": 0.92},
            feature_names=["age"],
            hyperparameters={},
            training_time=10.0,
        )
        entry2 = agent.register_model(
            model=model,
            model_name="TestModel",
            dataset_name="Heart Disease",
            problem_type="Classification",
            algorithm="XGBoost",
            metrics={"accuracy": 0.95},
            feature_names=["age"],
            hyperparameters={},
            training_time=15.0,
        )

        rolled = agent.rollback_model(entry1["model_id"])
        assert rolled["model_id"] == entry1["model_id"]
        assert rolled["is_active"] is True
        assert rolled["deployment_status"] == "Rolled Back"

        versions = agent.get_model_versions("TestModel")
        active_models = [m for m in versions if m["is_active"]]
        assert len(active_models) == 1
        assert active_models[0]["model_id"] == entry1["model_id"]


def test_generate_leaderboard_ranking():
    with tempfile.TemporaryDirectory() as tempdir:
        registry_path = os.path.join(tempdir, "model_registry.json")
        agent = ModelRegistryAgent(registry_path=registry_path)

        model = LogisticRegression()
        agent.register_model(
            model=model,
            model_name="ModelA",
            dataset_name="Heart Disease",
            problem_type="Classification",
            algorithm="Logistic Regression",
            metrics={"accuracy": 0.88, "precision": 0.85, "recall": 0.80, "f1": 0.82, "roc_auc": 0.87},
            feature_names=["age"],
            hyperparameters={},
            training_time=9.0,
        )
        agent.register_model(
            model=model,
            model_name="ModelB",
            dataset_name="Heart Disease",
            problem_type="Classification",
            algorithm="Random Forest",
            metrics={"accuracy": 0.92, "precision": 0.90, "recall": 0.88, "f1": 0.89, "roc_auc": 0.93},
            feature_names=["age"],
            hyperparameters={},
            training_time=14.0,
        )

        leaderboard = agent.generate_leaderboard("Classification")
        assert leaderboard[0]["model_name"] == "ModelB"
        assert leaderboard[0]["rank"] == 1
        assert leaderboard[1]["model_name"] == "ModelA"


if __name__ == "__main__":
    test_register_model_and_persist_registry()
    test_get_model_versions_and_compare_versions()
    test_rollback_model()
    test_generate_leaderboard_ranking()
    print("ALL model registry tests passed")

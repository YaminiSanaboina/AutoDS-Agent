import os
import sys
import tempfile

sys.path.insert(0, os.getcwd())

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from agents.retraining_agent import AutonomousRetrainingAgent


def test_high_drift_triggers_retraining():
    agent = AutonomousRetrainingAgent(history_path=os.path.join(tempfile.gettempdir(), "retrain_history1.json"))
    drift_report = {"severity": "High"}
    decision = agent.should_retrain(drift_report, current_model_performance={"performance_drop_pct": 0.0})
    assert decision["decision"] is True
    assert decision["priority"] == "Critical"
    print("PASS: High drift triggers retraining")


def test_low_drift_does_not_retrain():
    agent = AutonomousRetrainingAgent(history_path=os.path.join(tempfile.gettempdir(), "retrain_history2.json"))
    drift_report = {"severity": "Low"}
    decision = agent.should_retrain(drift_report, current_model_performance={"performance_drop_pct": 0.01})
    assert decision["decision"] is False
    assert decision["priority"] == "Low"
    print("PASS: Low drift does not trigger retraining")


def test_compare_models_recommends_deployment_when_improved():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=50, n_features=4, n_informative=2, n_redundant=0, random_state=42)
    old_model = DummyClassifier(strategy="most_frequent").fit(X, y)
    new_model = LogisticRegression(max_iter=1000).fit(X, y)
    agent = AutonomousRetrainingAgent(history_path=os.path.join(tempfile.gettempdir(), "retrain_history3.json"))
    result = agent.compare_models(old_model, new_model, pd.DataFrame(X), pd.Series(y), problem_type="Classification")
    assert result["deploy_new_model"] is True
    assert result["improvement"] > 0
    print("PASS: New model improves performance and is recommended for deployment")


def test_compare_models_keeps_old_model_when_worse():
    from sklearn.datasets import make_classification
    from sklearn.dummy import DummyClassifier

    X, y = make_classification(n_samples=50, n_features=4, n_informative=2, n_redundant=0, random_state=42)
    old_model = LogisticRegression(max_iter=1000).fit(X, y)
    new_model = DummyClassifier(strategy="most_frequent").fit(X, y)
    agent = AutonomousRetrainingAgent(history_path=os.path.join(tempfile.gettempdir(), "retrain_history4.json"))
    result = agent.compare_models(old_model, new_model, pd.DataFrame(X), pd.Series(y), problem_type="Classification")
    assert result["deploy_new_model"] is False
    print("PASS: Worse new model keeps the existing model")


def test_retraining_failure_generates_self_healing_recommendation():
    df = pd.DataFrame({"feature": [1, 2, 3], "label2": [0, 1, 0]})
    agent = AutonomousRetrainingAgent(history_path=os.path.join(tempfile.gettempdir(), "retrain_history5.json"))
    result = agent.retrain_model(
        df,
        target_column="missing_target",
        dataset_name="failure_dataset",
    )
    assert result["status"] == "failed"
    assert "recovery" in result
    assert result["recovery"].get("recommended_action") is not None
    print("PASS: Training failure returns a self-healing recommendation")


if __name__ == "__main__":
    test_high_drift_triggers_retraining()
    test_low_drift_does_not_retrain()
    test_compare_models_recommends_deployment_when_improved()
    test_compare_models_keeps_old_model_when_worse()
    test_retraining_failure_generates_self_healing_recommendation()
    print("ALL retraining agent tests passed")

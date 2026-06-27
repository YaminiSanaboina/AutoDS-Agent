import pandas as pd

from agents.ai_ethics_agent import AIEthicsAgent


def test_dataset_bias_detection():
    df = pd.DataFrame({
        "gender": ["male", "female", "female", "male", "female"],
        "age": [25, 35, 45, 55, 65],
        "score": [0, 1, 1, 0, 1],
    })
    agent = AIEthicsAgent()
    report = agent.analyze_dataset_bias(df)
    assert report["sensitive_features"]
    assert report["bias_risk"] in {"Low", "Medium", "High"}
    assert "gender" in [col.lower() for col in report["sensitive_features"]]


def test_privacy_risk_scanner():
    df = pd.DataFrame({
        "email": ["a@example.com", "b@example.com"],
        "phone": ["123", "456"],
        "age": [30, 40],
    })
    agent = AIEthicsAgent()
    report = agent.analyze_privacy_risk(df)
    assert report["privacy_risk"] == "High"
    assert "email" in [col.lower() for col in report["detected_identifiers"]]


def test_model_fairness_evaluation():
    agent = AIEthicsAgent()
    predictions = [0, 1, 1, 0, 1, 0]
    actual = [0, 1, 0, 0, 1, 1]
    sensitive_groups = {
        "group_a": [0, 1, 2],
        "group_b": [3, 4, 5],
    }
    report = agent.evaluate_model_fairness(predictions, actual, sensitive_groups)
    assert 0 <= report["fairness_score"] <= 100
    assert report["risk_level"] in {"Low", "Medium", "High"}
    assert "group_a" in report["group_metrics"]


def test_governance_score_generation():
    agent = AIEthicsAgent()
    bias_report = {"bias_risk": "Low"}
    fairness_report = {"risk_level": "Low"}
    privacy_report = {"privacy_risk": "Medium"}
    score = agent.calculate_ai_governance_score(bias_report, fairness_report, privacy_report)
    assert score["score"] <= 100
    assert score["grade"] in {"A", "B", "C", "D"}
    assert score["readiness"] in {"Production Ready", "Needs Review"}

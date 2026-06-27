import sys
import os
sys.path.insert(0, os.getcwd())

from utils.session_manager import init_session
from agents.self_healing_agent import SelfHealingAgent


def test_encoding_error_detection():
    init_session()
    agent = SelfHealingAgent()
    analysis = agent.analyze_error("ValueError: could not convert string to float", {"name": "Iris"})
    assert analysis["error_type"] == "Encoding Error"
    assert "encoding" in analysis["root_cause"].lower()
    fix = agent.recommend_fix(analysis)
    assert "one-hot" in fix["recommended_action"].lower() or "label encoding" in fix["recommended_action"].lower()
    print("PASS: Encoding error detected and fix recommended")


def test_missing_value_error_detection():
    init_session()
    agent = SelfHealingAgent()
    analysis = agent.analyze_error("Input contains NaN values in feature columns.", {"name": "Titanic"})
    assert analysis["error_type"] == "Missing Value Error"
    fix = agent.recommend_fix(analysis)
    assert "impute" in fix["recommended_action"].lower()
    print("PASS: Missing value error detected and fix recommended")


def test_memory_error_detection():
    init_session()
    agent = SelfHealingAgent()
    analysis = agent.analyze_error("MemoryError: Unable to allocate array for training.", {"name": "Housing"})
    assert analysis["error_type"] == "Memory Error"
    fix = agent.recommend_fix(analysis)
    assert "reduce" in fix["recommended_action"].lower() or "memory" in fix["recommended_action"].lower()
    print("PASS: Memory error detected and fix recommended")


def test_overfitting_detection_from_metrics():
    init_session()
    agent = SelfHealingAgent()
    analysis = agent.analyze_error(
        "Training score and test score indicate poor generalization.",
        {"training_score": 0.95, "test_score": 0.7},
    )
    assert analysis["error_type"] == "Overfitting"
    fix = agent.recommend_fix(analysis)
    assert "cross-validation" in fix["recommended_action"].lower() or "regularization" in fix["recommended_action"].lower()
    print("PASS: Overfitting warning detected from metrics")


def test_error_history_and_reports():
    init_session()
    agent = SelfHealingAgent()
    agent.analyze_error("ValueError: could not convert string to float", {"name": "Iris"})
    agent.analyze_error("Input contains NaN values in feature columns.", {"name": "Titanic"})
    history = agent.generate_health_report()
    assert history["total_failures"] >= 2
    assert history["most_common_issue"] in {"Encoding Error", "Missing Value Error", "Class Imbalance", "Memory Error", "Shape Mismatch", "Overfitting", "Unknown Error"}
    simulation = agent.simulate_fix()
    assert simulation["requires_user_confirmation"] is True
    plan = agent.get_recovery_plan()
    assert isinstance(plan["steps"], list)
    print("PASS: Error history, health report, simulation, and recovery plan work")


if __name__ == "__main__":
    test_encoding_error_detection()
    test_missing_value_error_detection()
    test_memory_error_detection()
    test_overfitting_detection_from_metrics()
    test_error_history_and_reports()
    print("ALL self-healing tests passed")

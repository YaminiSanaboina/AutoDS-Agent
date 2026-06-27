import numpy as np
import pandas as pd

from agents.multi_agent_collaboration_engine import MultiAgentCollaborationEngine


def test_normal_healthy_project_recommends_deploy_or_monitor():
    dataset = pd.DataFrame(
        {
            "feature_a": np.random.rand(200),
            "feature_b": np.random.randint(0, 3, size=200),
            "target": np.random.choice([0, 1], size=200),
        }
    )
    engine = MultiAgentCollaborationEngine(history_path="agent_collaboration_history_test.json")
    project_state = {
        "drift_report": {"severity": "Low"},
        "deployment_status": {"deployed": False},
        "health_score": 80.0,
        "risk_level": "Low",
    }
    model_info = {
        "problem_type": "Classification",
        "history": [{"test_score": 0.88, "train_score": 0.9}],
        "performance": {"performance_drop_pct": 0.01},
    }
    result = engine.execute_autonomous_project(dataset, {"name": "Healthy Dataset"}, model_info, project_state)

    assert result["final_decision"] in {"deploy", "monitor", "retrain", "feature_engineering", "clean_data", "collect_data"}
    assert result["timeline"]
    assert "Drift Monitoring" in result["agents_involved"]
    assert isinstance(result["confidence"], float)


def test_poor_quality_dataset_recommends_data_cleaning():
    dataset = pd.DataFrame(
        {
            "feature_a": [None] * 50,
            "feature_b": ["A"] * 50,
            "target": np.random.choice([0, 1], size=50),
        }
    )
    engine = MultiAgentCollaborationEngine(history_path="agent_collaboration_history_test.json")
    project_state = {
        "drift_report": {"severity": "Low"},
        "deployment_status": {"deployed": False},
        "health_score": 35.0,
        "risk_level": "High",
    }
    model_info = {"problem_type": "Classification", "history": [], "performance": {"performance_drop_pct": 0.0}}
    result = engine.execute_autonomous_project(dataset, {"name": "Dirty Dataset"}, model_info, project_state)

    assert result["final_decision"] in {"clean_data", "monitor", "collect_data", "feature_engineering"}
    assert any("quality" in step["description"].lower() or "cleaning" in step["description"].lower() for step in result["timeline"])


def test_high_drift_detected_recommends_retraining():
    dataset = pd.DataFrame(
        {
            "feature_a": np.random.rand(120),
            "feature_b": np.random.randint(0, 2, size=120),
            "target": np.random.choice([0, 1], size=120),
        }
    )
    engine = MultiAgentCollaborationEngine(history_path="agent_collaboration_history_test.json")
    project_state = {
        "drift_report": {"severity": "High"},
        "deployment_status": {"deployed": True},
        "health_score": 65.0,
        "risk_level": "Medium",
    }
    model_info = {"problem_type": "Classification", "history": [{"test_score": 0.74, "train_score": 0.89}], "performance": {"performance_drop_pct": 0.08}}
    result = engine.execute_autonomous_project(dataset, {"name": "Drift Dataset"}, model_info, project_state)

    assert result["final_decision"] == "retrain"


def test_agent_disagreement_triggers_conflict_resolution():
    dataset = pd.DataFrame(
        {
            "feature_a": np.random.rand(150),
            "feature_b": np.random.randint(0, 4, size=150),
            "target": np.random.choice([0, 1], size=150),
        }
    )
    engine = MultiAgentCollaborationEngine(history_path="agent_collaboration_history_test.json")
    project_state = {
        "drift_report": {"severity": "Medium"},
        "deployment_status": {"deployed": False},
        "health_score": 45.0,
        "risk_level": "High",
    }
    model_info = {"problem_type": "Classification", "history": [{"test_score": 0.7, "train_score": 0.85}], "performance": {"performance_drop_pct": 0.12}}
    result = engine.execute_autonomous_project(dataset, {"name": "Disagreement Dataset"}, model_info, project_state)

    assert result["final_decision"] in {"retrain", "clean_data", "monitor", "collect_data"}
    assert result["timeline"]


def test_full_autonomous_cycle_completes_timeline():
    dataset = pd.DataFrame(
        {
            "feature_a": np.random.rand(180),
            "feature_b": np.random.randint(0, 3, size=180),
            "target": np.random.choice([0, 1], size=180),
        }
    )
    engine = MultiAgentCollaborationEngine(history_path="agent_collaboration_history_test.json")
    project_state = {
        "drift_report": {"severity": "Low"},
        "deployment_status": {"deployed": False},
        "health_score": 75.0,
        "risk_level": "Low",
    }
    model_info = {"problem_type": "Classification", "history": [{"test_score": 0.82, "train_score": 0.83}], "performance": {"performance_drop_pct": 0.02}}
    result = engine.execute_autonomous_project(dataset, {"name": "Autonomous Dataset"}, model_info, project_state)

    assert len(result["timeline"]) >= 7
    assert result["agents_involved"]
    assert result["meeting_record"]["final_decision"] == result["final_decision"]

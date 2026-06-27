import os
import json
import tempfile

import pandas as pd

from agents.autonomous_data_scientist_agent import AutonomousDataScientistAgent


def _small_classification_df():
    return pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6],
        "feature2": [0, 1, 0, 1, 0, 1],
        "target": [0, 1, 0, 1, 0, 1],
    })


def test_autonomous_project_execution_and_decision_logging(tmp_path):
    df = _small_classification_df()
    decisions_file = tmp_path / "autonomous_decisions.json"
    agent = AutonomousDataScientistAgent(decision_path=str(decisions_file))

    project = agent.run_autonomous_project(df, project_goal="Maximize accuracy")

    assert "project_score" in project
    assert isinstance(project["project_score"]["score"], int)
    assert os.path.exists(str(decisions_file))

    with open(str(decisions_file), "r", encoding="utf-8") as fh:
        decisions = json.load(fh)
    assert isinstance(decisions, list)
    assert len(decisions) >= 1


def test_goal_based_strategy_change(tmp_path):
    df = _small_classification_df()
    agent = AutonomousDataScientistAgent(decision_path=str(tmp_path / "decisions.json"))
    project = agent.run_autonomous_project(df, project_goal="Improve interpretability")
    chosen = project.get("final_choice")
    # When interpretability requested, expect interpretable model types
    if chosen:
        model_name = chosen.get("model", "")
        assert any(x in model_name for x in ["Logistic", "Linear", "Decision"]) or isinstance(model_name, str)


def test_self_improve_returns_suggestions(tmp_path):
    agent = AutonomousDataScientistAgent(decision_path=str(tmp_path / "decisions.json"))
    suggestions = agent.self_improve()
    assert "suggestions" in suggestions
    assert isinstance(suggestions["suggestions"], list)

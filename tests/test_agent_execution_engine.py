import os
import json
import pandas as pd

from agents.agent_execution_engine import AgentExecutionEngine


def test_execute_command_analyze(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0, 1, 0], "target": [0, 1, 0]})
    engine = AgentExecutionEngine(history_path=str(tmp_path / "history.json"))
    res = engine.execute_command("Please analyze my data", context={"dataset": df, "dataset_name": "test"})
    assert res.get("intent") == "analyze_data"
    assert "pipeline" in res
    assert os.path.exists(str(tmp_path / "history.json"))
    assert isinstance(res.get("confidence"), float)


def test_run_pipeline_tracking(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 0, 1]})
    engine = AgentExecutionEngine(history_path=str(tmp_path / "history.json"))
    plan = [{"step": 1, "action": "Feature Engineering"}, {"step": 2, "action": "Run Dataset Intelligence analysis"}]
    results = engine.run_pipeline(plan, context={"dataset": df, "dataset_name": "demo"})
    assert isinstance(results, list)
    assert all("execution_time" in item for item in results)
    assert any(item["status"] == "SUCCESS" for item in results)


def test_failure_handling_and_self_healing(tmp_path, monkeypatch):
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"], "target": [0, 1, 0]})
    engine = AgentExecutionEngine(history_path=str(tmp_path / "history.json"))

    # Monkeypatch _execute_action to raise for first action to simulate failure
    original_execute = engine._execute_action

    def failing_execute(action, context):
        if "Feature" in action:
            raise RuntimeError("simulated failure: encoding")
        return original_execute(action, context)

    monkeypatch.setattr(engine, "_execute_action", failing_execute)

    res = engine.execute_command("Improve accuracy", context={"dataset": df, "dataset_name": "demo"})
    # ensure pipeline recorded and history persisted
    assert os.path.exists(str(tmp_path / "history.json"))
    with open(str(tmp_path / "history.json"), "r", encoding="utf-8") as fh:
        hist = json.load(fh)
    assert isinstance(hist, list)
    # pipeline should contain failed step (or retried)
    pipeline = res.get("pipeline")
    assert isinstance(pipeline, list)
    assert any("feature" in (item.get("step") or "").lower() for item in pipeline)

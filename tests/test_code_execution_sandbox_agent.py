import json
import os
import pandas as pd

from agents.code_execution_sandbox_agent import CodeExecutionSandboxAgent


def test_valid_pandas_code_execution(tmp_path):
    agent = CodeExecutionSandboxAgent(history_path=str(tmp_path / "sandbox_history.json"))
    df = pd.DataFrame({"age": [20, 30], "income": [50, 80]})
    code = 'df["double_age"] = df["age"] * 2\nprint(df["double_age"].tolist())'
    result = agent.execute_code(code, context={"dataset": df, "dataset_name": "demo"})
    assert result["success"] is True
    assert "[40, 60]" in result["output"]
    assert os.path.exists(str(tmp_path / "sandbox_history.json"))


def test_dangerous_code_detection(tmp_path):
    agent = CodeExecutionSandboxAgent(history_path=str(tmp_path / "sandbox_history.json"))
    code = 'import os\nos.remove("data.csv")'
    result = agent.execute_code(code)
    assert result["success"] is False
    assert result["error"] == "Unsafe operation detected"
    assert "Unsafe operation detected" in result["suggested_fix"] or result["error_type"] == "UnsafeCode"


def test_error_handling_with_self_healing(tmp_path):
    agent = CodeExecutionSandboxAgent(history_path=str(tmp_path / "sandbox_history.json"))
    df = pd.DataFrame({"x": [1, 2]})
    code = 'df["y"] = unknown_variable'
    result = agent.execute_code(code, context={"dataset": df, "dataset_name": "demo"})
    assert result["success"] is False
    assert result["error_type"] is not None
    assert result["suggested_fix"] is not None


def test_execution_history_saving(tmp_path):
    history_file = tmp_path / "sandbox_history.json"
    agent = CodeExecutionSandboxAgent(history_path=str(history_file))
    df = pd.DataFrame({"age": [1]})
    code = 'df["double_age"] = df["age"] * 2'
    result = agent.execute_code(code, context={"dataset": df, "dataset_name": "demo"})
    assert history_file.exists()
    with open(str(history_file), "r", encoding="utf-8") as fh:
        history = json.load(fh)
    assert isinstance(history, list)
    assert history[-1]["success"] == result["success"]

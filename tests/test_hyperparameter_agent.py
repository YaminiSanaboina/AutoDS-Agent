import os
import sys
import tempfile

sys.path.insert(0, os.getcwd())

import pandas as pd

from agents.hyperparameter_agent import HyperparameterOptimizationAgent


def test_optimize_logistic_regression():
    with tempfile.TemporaryDirectory() as tempdir:
        memory_path = os.path.join(tempdir, "hyperparam_memory.json")
        agent = HyperparameterOptimizationAgent(memory_path=memory_path)
        df = pd.DataFrame(
            {
                "feature_1": [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4],
                "feature_2": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )
        result = agent.optimize(
            X=df[["feature_1", "feature_2"]],
            y=df["target"],
            model_name="Logistic Regression",
            problem_type="Classification",
            cv=2,
            use_history=False,
        )

        assert result["status"] == "success"
        assert "best_params" in result
        assert "optimized_score" in result
        assert result["optimized_score"] >= 0.0
        print("PASS: Logistic Regression hyperparameter optimization")


def test_unsupported_model():
    agent = HyperparameterOptimizationAgent()
    result = agent.optimize(
        X=pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
        y=pd.Series([0, 1, 0]),
        model_name="Unsupported Model",
        problem_type="Classification",
        use_history=False,
    )
    assert result["status"] == "failed"
    assert "Unsupported model" in result["diagnostics"]
    print("PASS: Unsupported model handled gracefully")


def test_compare_before_after():
    agent = HyperparameterOptimizationAgent()
    comparison = agent.compare_before_after(0.70, 0.80, "Random Forest")
    assert comparison["recommended_model"] == "Random Forest"
    assert comparison["improvement"] == "10.0%"
    print("PASS: Comparison before and after returns improvement details")


if __name__ == "__main__":
    test_optimize_logistic_regression()
    test_unsupported_model()
    test_compare_before_after()
    print("ALL hyperparameter agent tests passed")

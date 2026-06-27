import sys
import os
import json
import tempfile

sys.path.insert(0, os.getcwd())

from agents.experiment_memory_agent import ExperimentMemoryAgent


def test_log_and_history():
    with tempfile.TemporaryDirectory() as tempdir:
        storage = os.path.join(tempdir, "memory.json")
        agent = ExperimentMemoryAgent(storage)
        agent.log_experiment(
            dataset_name="Housing.csv",
            dataset_shape=[100, 10],
            problem_type="Regression",
            algorithm_name="Random Forest",
            hyperparameters={"n_estimators": 100},
            train_score=0.95,
            test_score=0.92,
            cv_score=0.9,
            training_time=120.0,
            feature_count=10,
            feature_engineering_steps=["price_per_area"],
            data_cleaning_steps=["median imputation"],
            notes="Good baseline.",
        )
        agent.log_experiment(
            dataset_name="Housing.csv",
            algorithm_name="XGBoost",
            train_score=0.96,
            test_score=0.93,
            cv_score=0.91,
        )
        history_all = agent.get_history()
        history_filtered = agent.get_history(dataset_name="Housing.csv", algorithm_name="Random Forest")
        assert len(history_all) == 2
        assert len(history_filtered) == 1
        print("PASS: Log multiple experiments and retrieve history")


def test_best_experiment():
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "memory.json")
        agent = ExperimentMemoryAgent(path)
        agent.log_experiment("Titanic.csv", algorithm_name="Logistic Regression", train_score=0.85, test_score=0.82)
        agent.log_experiment("Titanic.csv", algorithm_name="Random Forest", train_score=0.9, test_score=0.88)
        best = agent.get_best_experiment()
        assert best["best_experiment"]["algorithm_name"] == "Random Forest"
        assert "Highest test score" in best["reasons"][0]
        print("PASS: Best experiment found")


def test_compare_experiments():
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "memory.json")
        agent = ExperimentMemoryAgent(path)
        agent.log_experiment("Customer.csv", algorithm_name="Logistic Regression", test_score=0.78)
        agent.log_experiment("Customer.csv", algorithm_name="XGBoost", test_score=0.85)
        comparison = agent.compare_experiments(dataset_name="Customer.csv")
        assert "Logistic Regression" in comparison
        assert "XGBoost" in comparison
        print("PASS: Model comparison intelligence")


def test_record_failure_and_recommendation():
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "memory.json")
        agent = ExperimentMemoryAgent(path)
        agent.record_failure(
            dataset_name="Heart Disease",
            algorithm_name="XGBoost",
            error_message="ValueError: could not convert string to float",
            failure_reason="Missing encoding for categorical input.",
            suggested_solution="Apply one-hot encoding before training.",
        )
        rec = agent.recommend_next_experiment()
        assert "Fix latest failure" in rec["recommendation"][0]
        print("PASS: Failure recording and optimization advisor")


def test_persistence_and_corruption_handling():
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "memory.json")
        agent = ExperimentMemoryAgent(path)
        agent.log_experiment("Housing.csv", algorithm_name="Random Forest", test_score=0.9)
        assert os.path.exists(path)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("INVALID JSON")
        agent2 = ExperimentMemoryAgent(path)
        assert agent2.get_history() == []
        print("PASS: JSON persistence and corrupted file handling")


if __name__ == "__main__":
    test_log_and_history()
    test_best_experiment()
    test_compare_experiments()
    test_record_failure_and_recommendation()
    test_persistence_and_corruption_handling()
    print("ALL experiment memory tests passed")

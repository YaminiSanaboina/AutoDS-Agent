import pandas as pd

from agents.synthetic_data_agent import SyntheticDataAgent


def test_statistical_generation_preserves_columns_and_count():
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "gender": ["male", "female", "female", "male"],
    })
    agent = SyntheticDataAgent()
    result = agent.generate_synthetic_data(df, num_samples=10, method="statistical")
    synthetic_df = result["synthetic_df"]

    assert synthetic_df.shape == (10, 2)
    assert list(synthetic_df.columns) == list(df.columns)
    assert result["method"] == "statistical"
    assert 0 <= result["quality_score"] <= 100


def test_bootstrap_generation_shape():
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": ["a", "b", "c"],
    })
    agent = SyntheticDataAgent()
    result = agent.generate_synthetic_data(df, num_samples=6, method="bootstrap")
    assert result["synthetic_df"].shape == (6, 2)
    assert result["method"] == "bootstrap"


def test_similarity_score_range():
    df = pd.DataFrame({
        "num": [1, 2, 3, 4, 5],
        "cat": ["a", "b", "a", "b", "c"],
    })
    agent = SyntheticDataAgent()
    synthetic = agent.generate_synthetic_data(df, num_samples=5, method="statistical")["synthetic_df"]
    sim = agent.evaluate_similarity(df, synthetic)
    assert 0 <= sim["similarity_score"] <= 100
    assert isinstance(sim["numeric_similarity"], dict)
    assert isinstance(sim["categorical_similarity"], dict)


def test_privacy_evaluation_detects_duplicates():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "value": [10, 20, 30],
    })
    agent = SyntheticDataAgent()
    synthetic = df.copy()
    report = agent.calculate_privacy_score(df, synthetic)
    assert report["privacy_score"] < 100
    assert report["risk_level"] in {"Low", "Medium", "High"}
    assert any("duplicate" in issue.lower() for issue in report["issues"]) or any("identifier" in issue.lower() for issue in report["issues"])


def test_augmentation_strategy_small_dataset():
    df = pd.DataFrame({
        "feature": [1, 2, 3],
        "target": [0, 1, 0],
    })
    agent = SyntheticDataAgent()
    recommendation = agent.recommend_augmentation_strategy(df)
    assert recommendation["recommended_method"] in {"statistical", "balanced", "bootstrap"}
    assert recommendation["reasoning"]
    assert "synthetic" in recommendation["reasoning"].lower() or "balanced" in recommendation["reasoning"].lower()

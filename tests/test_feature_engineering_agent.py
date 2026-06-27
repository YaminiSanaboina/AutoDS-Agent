import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
from agents.feature_engineering_agent import FeatureEngineeringAgent


def test_housing_feature_recommendations():
    df = pd.DataFrame({
        "area": [1500, 2000, 1800],
        "price": [300000, 420000, 360000],
        "bedrooms": [3, 4, 3],
        "bathrooms": [2, 3, 2],
    })
    agent = FeatureEngineeringAgent(df, name="Housing.csv")
    analysis = agent.analyze_features()
    assert analysis["area"]["type"] == "Numerical"
    suggestions = agent.suggest_new_features()
    assert "price_per_area" in suggestions
    assert "room_density" in suggestions
    print("PASS: Housing feature detection and area-related suggestions")


def test_titanic_feature_recommendations():
    df = pd.DataFrame({
        "name": ["Braund, Mr. Owen Harris", "Cumings, Mrs. John Bradley", "Heikkinen, Miss. Laina"],
        "sibsp": [1, 1, 0],
        "parch": [0, 0, 0],
        "sex": ["male", "female", "female"],
        "age": [22.0, 38.0, 26.0],
        "survived": [0, 1, 1],
    })
    agent = FeatureEngineeringAgent(df, name="Titanic-Dataset.csv")
    encoding = agent.recommend_encoding()
    assert encoding["sex"] == "One-Hot Encoding"
    suggestions = agent.suggest_new_features()
    assert "family_size" in suggestions
    assert "title" in suggestions
    print("PASS: Titanic categorical encoding and family size recommendation")


def test_heart_disease_feature_recommendations():
    df = pd.DataFrame({
        "age": [63, 37, 41],
        "sex": [1, 1, 0],
        "blood_pressure": [145, 130, 130],
        "cholesterol": [233, 250, 204],
        "target": [1, 1, 0],
    })
    agent = FeatureEngineeringAgent(df, name="Heart Disease")
    suggestions = agent.suggest_new_features()
    assert "age_group" in suggestions
    assert "health_risk_score" in suggestions
    print("PASS: Heart Disease healthcare-specific feature suggestions")


def test_missing_values_strategy():
    df = pd.DataFrame({
        "age": [25.0, None, 45.0],
        "gender": ["male", None, "female"],
        "signup_date": ["2021-01-01", "2021-02-05", None],
    })
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    agent = FeatureEngineeringAgent(df, name="Generic.csv")
    strategies = agent.recommend_missing_value_strategy()
    assert strategies["age"] == "Fill missing values using median"
    assert strategies["gender"] == "Fill missing values using mode"
    assert strategies["signup_date"] == "Fill missing values using forward fill or domain-specific handling"
    print("PASS: Missing value strategy recommendations")


def test_feature_plan_generation():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "area": [1500, 2000, 1800],
        "price": [300000, 420000, 360000],
        "gender": ["male", "female", "female"],
    })
    agent = FeatureEngineeringAgent(df, name="Housing.csv")
    plan = agent.generate_feature_plan()
    assert plan["current_features"] == 4
    assert "Encode categorical variables" in plan["steps"]
    print("PASS: Feature plan generation")


if __name__ == "__main__":
    test_housing_feature_recommendations()
    test_titanic_feature_recommendations()
    test_heart_disease_feature_recommendations()
    test_missing_values_strategy()
    test_feature_plan_generation()
    print("ALL feature engineering tests passed")

"""Tests for beginner-friendly dataset explanations via the AI assistant."""

from __future__ import annotations

import pandas as pd

from tests.conftest import set_test_dataset


def test_iris_without_species(ai_assistant, session_ready):
    iris_df = pd.DataFrame(
        {
            "sepal length": [5.1, 4.9, 5.8],
            "sepal width": [3.5, 3.0, 2.7],
            "petal length": [1.4, 1.4, 4.1],
            "petal width": [0.2, 0.2, 1.0],
        }
    )
    set_test_dataset(iris_df, name="Iris.csv")
    response = ai_assistant.generate_response("Explain my dataset like I am a beginner")

    assert "Iris.csv" in response
    assert "species" not in response.lower()


def test_titanic(ai_assistant, session_ready):
    titanic_df = pd.DataFrame(
        {
            "age": [22.0, 38.0, 26.0],
            "sex": ["male", "female", "male"],
            "pclass": [3, 1, 3],
            "fare": [7.25, 71.2833, 7.925],
            "survived": [0, 1, 1],
        }
    )
    set_test_dataset(titanic_df, name="Titanic-Dataset.csv")
    response = ai_assistant.generate_response("What does this dataset represent?")

    assert "Titanic-Dataset.csv" in response
    assert "3 rows" in response or "rows" in response.lower()


def test_housing(ai_assistant, session_ready):
    housing_df = pd.DataFrame(
        {
            "area": [1500, 2000, 1800],
            "bedrooms": [3, 4, 3],
            "bathrooms": [2, 2.5, 2],
            "parking": [1, 2, 1],
            "price": [150000, 250000, 180000],
        }
    )
    set_test_dataset(housing_df, name="Housing.csv")
    response = ai_assistant.generate_response("Explain this data simply")

    assert "Housing.csv" in response
    assert "columns" in response.lower()


def test_wine(ai_assistant, session_ready):
    wine_df = pd.DataFrame(
        {
            "alcohol": [8.4, 9.0, 8.5],
            "acidity": [0.56, 0.76, 0.65],
            "citric acid": [0.28, 0.28, 0.30],
            "quality": [5, 6, 5],
        }
    )
    set_test_dataset(wine_df, name="Wine-Quality.csv")
    response = ai_assistant.generate_response("Explain my dataset like I am a beginner")

    assert "Wine-Quality.csv" in response


def test_unknown_dataset(ai_assistant, session_ready):
    generic_df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3],
            "feature_b": [4, 5, 6],
            "feature_c": [7, 8, 9],
            "target": [0, 1, 0],
        }
    )
    set_test_dataset(generic_df, name="custom_data.csv")
    response = ai_assistant.generate_response("Explain this data simply")

    assert "custom_data.csv" in response


def count_sentences(text: str) -> int:
    return text.count(".") + text.count("!") + text.count("?")


def test_response_length(ai_assistant, session_ready):
    iris_df = pd.DataFrame(
        {
            "sepal length": [5.1, 4.9],
            "sepal width": [3.5, 3.0],
            "petal length": [1.4, 1.4],
            "petal width": [0.2, 0.2],
        }
    )
    set_test_dataset(iris_df, name="Iris.csv")
    response = ai_assistant.generate_response("Explain my dataset like I am a beginner")

    sentences = count_sentences(response)
    assert 2 <= sentences <= 10

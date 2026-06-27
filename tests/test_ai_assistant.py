"""Smoke tests for the legacy AI assistant response generator."""

from __future__ import annotations

import pandas as pd

from tests.conftest import set_test_dataset


def test_generate_response_returns_dataset_context(ai_assistant, session_ready):
    iris_df = pd.DataFrame(
        {
            "sepal length": [5.1, 4.9, 5.8],
            "sepal width": [3.5, 3.0, 2.7],
            "petal length": [1.4, 1.4, 4.1],
            "petal width": [0.2, 0.2, 1.0],
        }
    )
    set_test_dataset(iris_df, name="Iris.csv")

    prompts = [
        "Explain my dataset like I am a beginner",
        "What does this dataset represent?",
        "Explain this data simply",
    ]

    for prompt in prompts:
        response = ai_assistant.generate_response(prompt)
        assert isinstance(response, str)
        assert response.strip()
        assert "Iris.csv" in response
        assert "rows" in response.lower()
        assert "columns" in response.lower()


def test_generate_response_handles_empty_prompt(ai_assistant, session_ready):
    response = ai_assistant.generate_response("   ")
    assert "ask a question" in response.lower()

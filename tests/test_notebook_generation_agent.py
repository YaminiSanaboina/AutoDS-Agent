import json

from agents.notebook_generation_agent import NotebookGenerationAgent


def test_generate_notebook_structure():
    agent = NotebookGenerationAgent()
    notebook = agent.generate_notebook(
        project_title="AutoDS Demo",
        dataset_name="Heart Disease",
        target_column="target",
        problem_type="Classification",
        feature_names=["age", "cholesterol"],
        model_choice="Logistic Regression",
        dataset_description="A diagnostics dataset for heart disease prediction.",
    )

    assert notebook["nbformat"] == 4
    assert "cells" in notebook
    assert len(notebook["cells"]) > 0
    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert "AutoDS Demo" in notebook["cells"][0]["source"][0]


def test_generate_notebook_code_references_target_column():
    agent = NotebookGenerationAgent()
    notebook = agent.generate_notebook(
        project_title="AutoDS Demo",
        dataset_name="Titanic",
        target_column="Survived",
        problem_type="Classification",
        feature_names=["Pclass", "Sex"],
        model_choice="Random Forest",
    )

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert any("df = pd.read_csv('data.csv')" in cell["source"][0] for cell in code_cells)
    assert any("df.drop(columns=['Survived'])" in cell["source"][0] for cell in code_cells)
    assert any("RandomForestClassifier()" in cell["source"][0] for cell in code_cells)


def test_notebook_serialization():
    agent = NotebookGenerationAgent()
    notebook = agent.generate_notebook(
        project_title="AutoDS Demo",
        dataset_name="Iris",
        target_column="species",
        problem_type="Classification",
        feature_names=["sepal_length", "sepal_width"],
        model_choice="Logistic Regression",
    )

    json_text = agent.to_json(notebook)
    parsed = json.loads(json_text)
    assert parsed["nbformat"] == 4
    assert parsed["cells"][0]["cell_type"] == "markdown"

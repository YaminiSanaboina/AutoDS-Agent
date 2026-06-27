import json

from agents.code_generation_agent import CodeGenerationAgent


def test_generate_complete_pipeline_classification():
    agent = CodeGenerationAgent()
    code = agent.generate_complete_pipeline(
        dataset_path="data/Heart-Disease.csv",
        target_column="target",
        problem_type="Classification",
        model_choice="Logistic Regression",
        output_model_path="model.joblib",
    )

    assert "def main()" in code
    assert "train_test_split" in code
    assert "LogisticRegression()" in code
    assert "save_model(model, 'model.joblib')" in code
    assert "accuracy_score" in code


def test_generate_complete_pipeline_regression():
    agent = CodeGenerationAgent()
    code = agent.generate_complete_pipeline(
        dataset_path="data/Housing.csv",
        target_column="SalePrice",
        problem_type="Regression",
        model_choice="Linear Regression",
    )

    assert "LinearRegression()" in code
    assert "r2_score" in code
    assert "mean_squared_error" in code


def test_generate_xai_code_includes_shap_and_features():
    agent = CodeGenerationAgent()
    code = agent.generate_xai_code(
        model_type="Logistic Regression",
        feature_names=["age", "cholesterol"],
        output_path="shap.png",
    )

    assert "shap.summary_plot" in code
    assert "shap.Explainer" not in code or "shap.Explainer" in code
    assert "'age'" in code
    assert "'cholesterol'" in code
    assert "plt.savefig('shap.png')" in code


def test_generate_api_code_validates_features_and_model_file():
    agent = CodeGenerationAgent()
    code = agent.generate_api_code(feature_names=["age", "cholesterol"], model_file="best_model.joblib")

    assert "@app.post('/predict')" in code
    assert "joblib.load('best_model.joblib')" in code
    assert "validate_payload" in code
    assert "'age'" in code
    assert "'cholesterol'" in code


def test_generate_requirements_returns_list():
    agent = CodeGenerationAgent()
    requirements = agent.generate_requirements()

    assert isinstance(requirements, list)
    assert "fastapi>=0.104.0" in requirements
    assert "pandas>=2.0.0" in requirements

import json
from typing import Any, Dict, List, Optional


class NotebookGenerationAgent:
    """Generates Jupyter notebooks for AutoDS projects."""

    def generate_notebook(
        self,
        project_title: str,
        dataset_name: str,
        target_column: str,
        problem_type: str,
        feature_names: List[str],
        model_choice: str,
        dataset_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        notebook: Dict[str, Any] = {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3.x",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

        notebook["cells"].extend([
            self._markdown_cell(f"# {project_title}"),
            self._markdown_cell("## Project Overview"),
            self._markdown_cell(
                "This notebook guides the AutoDS project through data loading, cleaning, modeling, evaluation, explainability, and business conclusions."
            ),
            self._markdown_cell("## Dataset Summary"),
            self._markdown_cell(
                dataset_description or f"Dataset: {dataset_name}. Target: {target_column}. Problem type: {problem_type}."
            ),
            self._markdown_cell("## Data Loading"),
            self._code_cell(
                "import pandas as pd\n"
                "df = pd.read_csv('data.csv')\n"
                "df.head()\n"
            ),
            self._markdown_cell("## Data Cleaning"),
            self._code_cell(
                "df = df.drop_duplicates().reset_index(drop=True)\n"
                "for col in df.columns:\n"
                "    if df[col].dtype == 'object':\n"
                "        df[col] = df[col].fillna('Unknown')\n"
                "    else:\n"
                "        df[col] = df[col].fillna(df[col].median())\n"
                "df.info()\n"
            ),
            self._markdown_cell("## Exploratory Data Analysis"),
            self._code_cell(
                "import seaborn as sns\n"
                "import matplotlib.pyplot as plt\n"
                "sns.pairplot(df.sample(min(5, len(df))))\n"
                "plt.show()\n"
            ),
            self._markdown_cell("## Feature Engineering"),
            self._code_cell(
                "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
                "from sklearn.compose import ColumnTransformer\n"
                "from sklearn.pipeline import Pipeline\n"
                f"X = df.drop(columns=['{target_column}'])\n"
                f"y = df['{target_column}']\n"
                "categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()\n"
                "numeric_columns = X.select_dtypes(include=['number']).columns.tolist()\n"
                "preprocessor = ColumnTransformer([\n"
                "    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),\n"
                "    ('num', StandardScaler(), numeric_columns),\n"
                "])\n"
                "pipeline = Pipeline([('preprocessor', preprocessor)])\n"
                "X_transformed = pipeline.fit_transform(X)\n"
                "X_transformed.shape\n"
            ),
            self._markdown_cell("## Model Training"),
            self._code_cell(
                "from sklearn.model_selection import train_test_split\n"
                "from joblib import dump\n"
                f"from sklearn.{self._model_module(model_choice, problem_type)} import {self._model_class(model_choice, problem_type)}\n"
                "X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)\n"
                f"model = {self._model_class(model_choice, problem_type)}()\n"
                "model.fit(X_train, y_train)\n"
                "dump(model, 'model.joblib')\n"
            ),
            self._markdown_cell("## Model Evaluation"),
            self._code_cell(
                "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_squared_error\n"
                "y_pred = model.predict(X_test)\n"
                "print('Prediction sample:', y_pred[:5])\n"
            ),
            self._markdown_cell("## Explainability"),
            self._code_cell(
                "import shap\n"
                "explainer = shap.Explainer(model, X_train)\n"
                "shap_values = explainer(X_test)\n"
                "shap.summary_plot(shap_values, X_test, show=False)\n"
            ),
            self._markdown_cell("## Conclusions"),
            self._markdown_cell(
                "Summarize key findings, model performance, and recommended next steps for deployment or retraining."
            ),
        ])

        return notebook

    def _markdown_cell(self, text: str) -> Dict[str, Any]:
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [text],
        }

    def _code_cell(self, code: str) -> Dict[str, Any]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [code],
        }

    def _model_class(self, model_choice: str, problem_type: str) -> str:
        if problem_type.title() == "Classification":
            return {
                "Logistic Regression": "LogisticRegression",
                "Random Forest": "RandomForestClassifier",
                "XGBoost": "XGBClassifier",
                "SVM": "SVC",
            }.get(model_choice, "LogisticRegression")
        return {
            "Linear Regression": "LinearRegression",
            "Random Forest Regressor": "RandomForestRegressor",
            "Gradient Boosting Regressor": "GradientBoostingRegressor",
        }.get(model_choice, "LinearRegression")

    def _model_module(self, model_choice: str, problem_type: str) -> str:
        if problem_type.title() == "Classification":
            if model_choice == "XGBoost":
                return "ensemble"
            return "linear_model" if model_choice == "Logistic Regression" else "ensemble"
        return "linear_model" if model_choice == "Linear Regression" else "ensemble"

    def to_json(self, notebook: Dict[str, Any]) -> str:
        return json.dumps(notebook, indent=2)

from typing import Any, Dict, List


class CodeGenerationAgent:
    """Generates production-ready Python code for AutoDS workflows."""

    DEFAULT_REQUIREMENTS = [
        "pandas>=2.0.0",
        "numpy>=1.25.0",
        "scikit-learn>=1.4.0",
        "joblib>=1.4.0",
        "shap>=0.42.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=1.10.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
    ]

    CLASSIFICATION_MODELS = {
        "Logistic Regression": "LogisticRegression",
        "Random Forest": "RandomForestClassifier",
        "XGBoost": "XGBClassifier",
        "SVM": "SVC",
    }

    REGRESSION_MODELS = {
        "Linear Regression": "LinearRegression",
        "Random Forest Regressor": "RandomForestRegressor",
        "Gradient Boosting Regressor": "GradientBoostingRegressor",
    }

    def generate_complete_pipeline(
        self,
        dataset_path: str,
        target_column: str,
        problem_type: str,
        model_choice: str,
        output_model_path: str = "model.joblib",
    ) -> str:
        """Generate a complete training pipeline script."""
        problem_type = problem_type.title()
        if problem_type not in {"Classification", "Regression"}:
            raise ValueError("problem_type must be either 'Classification' or 'Regression'.")

        imports = [
            "import logging",
            "import joblib",
            "import pandas as pd",
            "from sklearn.compose import ColumnTransformer",
            "from sklearn.pipeline import Pipeline",
            "from sklearn.model_selection import train_test_split",
            "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error, confusion_matrix",
            "from sklearn.preprocessing import OneHotEncoder, StandardScaler",
        ]

        if problem_type == "Classification":
            imports.extend([
                "from sklearn.linear_model import LogisticRegression",
                "from sklearn.ensemble import RandomForestClassifier",
                "from sklearn.svm import SVC",
                "from xgboost import XGBClassifier",
            ])
            model_class = self.CLASSIFICATION_MODELS.get(model_choice, "LogisticRegression")
        else:
            imports.extend([
                "from sklearn.linear_model import LinearRegression",
                "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor",
                "from xgboost import XGBRegressor",
            ])
            model_class = self.REGRESSION_MODELS.get(model_choice, "LinearRegression")

        constructor = f"{model_class}()"
        if problem_type == "Classification" and model_choice == "XGBoost":
            constructor = "XGBClassifier(use_label_encoder=False, eval_metric='logloss')"

        code_lines = [
            "# Auto-generated AutoDS pipeline",
            "",
            "def load_data(path: str):",
            "    df = pd.read_csv(path)",
            "    return df",
            "",
            "def clean_data(df):",
            "    df = df.drop_duplicates().reset_index(drop=True)",
            "    for column in df.columns:",
            "        if df[column].dtype == 'object':",
            "            df[column] = df[column].fillna('Unknown')",
            "        else:",
            "            df[column] = df[column].fillna(df[column].median())",
            "    return df",
            "",
            "def engineer_features(df, target_column):",
            "    X = df.drop(columns=[target_column])",
            "    y = df[target_column]",
            "    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()",
            "    numeric_columns = X.select_dtypes(include=['number']).columns.tolist()",
            "    transformers = []",
            "    if categorical_columns:",
            "        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns))",
            "    if numeric_columns:",
            "        transformers.append(('num', StandardScaler(), numeric_columns))",
            "    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')",
            "    pipeline = Pipeline([('preprocessor', preprocessor)])",
            "    X_processed = pipeline.fit_transform(X)",
            "    return X_processed, y",
            "",
            "def train_model(X, y):",
            f"    model = {constructor}",
            "    model.fit(X, y)",
            "    return model",
            "",
            "def evaluate_model(model, X_test, y_test):",
            "    y_pred = model.predict(X_test)",
        ]

        if problem_type == "Classification":
            code_lines.extend([
                "    accuracy = accuracy_score(y_test, y_pred)",
                "    precision = precision_score(y_test, y_pred, zero_division=0)",
                "    recall = recall_score(y_test, y_pred, zero_division=0)",
                "    f1 = f1_score(y_test, y_pred, zero_division=0)",
                "    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, 'predict_proba') else None",
                "    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc}",
            ])
        else:
            code_lines.extend([
                "    r2 = r2_score(y_test, y_pred)",
                "    mae = mean_absolute_error(y_test, y_pred)",
                "    rmse = mean_squared_error(y_test, y_pred, squared=False)",
                "    return {'r2': r2, 'mae': mae, 'rmse': rmse}",
            ])

        code_lines.extend([
            "",
            "def save_model(model, path: str):",
            "    joblib.dump(model, path)",
            "",
            "def main():",
            f"    dataset_path = '{dataset_path}'",
            f"    target_column = '{target_column}'",
            "    df = load_data(dataset_path)",
            "    df = clean_data(df)",
            "    X, y = engineer_features(df, target_column)",
            "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
            "    model = train_model(X_train, y_train)",
            "    evaluation = evaluate_model(model, X_test, y_test)",
            "    print('Evaluation:', evaluation)",
            f"    save_model(model, '{output_model_path}')",
            "",
            "if __name__ == '__main__':",
            "    main()",
        ])

        return "\n".join(imports + ["", *code_lines])

    def generate_xai_code(self, model_type: str, feature_names: List[str], output_path: str = "shap_summary.png") -> str:
        feature_list = ", ".join([f"'{name}'" for name in feature_names])
        model_type = model_type.title()
        explainer_line = "    explainer = shap.TreeExplainer(model)" if model_type in {"Random Forest", "Xgboost", "Random Forest Regressor", "Gradient Boosting Regressor"} else "    explainer = shap.LinearExplainer(model, X)"

        return "\n".join([
            "import joblib",
            "import shap",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "",
            "def load_model(path: str):",
            "    return joblib.load(path)",
            "",
            "def load_data(path: str):",
            "    return pd.read_csv(path)",
            "",
            "def explain_model(model, X):",
            "    shap.initjs()",
            explainer_line,
            "    shap_values = explainer(X)",
            "    shap.summary_plot(shap_values, X, show=False)",
            "    plt.tight_layout()",
            f"    plt.savefig('{output_path}')",
            "    plt.close()",
            "",
            "def main():",
            "    model = load_model('model.joblib')",
            "    data = load_data('data.csv')",
            f"    X = data[[{feature_list}]] if [{feature_list}] else data",
            "    explain_model(model, X)",
            "",
            "if __name__ == '__main__':",
            "    main()",
        ])

    def generate_api_code(self, feature_names: List[str], model_file: str = "model.joblib") -> str:
        fields = "\n".join([f"        '{name}'," for name in feature_names])
        return "\n".join([
            "from fastapi import FastAPI, HTTPException",
            "from pydantic import BaseModel",
            "import joblib",
            "import pandas as pd",
            "from typing import Any, Dict",
            "",
            "class PredictRequest(BaseModel):",
            "    features: Dict[str, Any]",
            "",
            "app = FastAPI(title='AutoDS Prediction API')",
            "",
            "try:",
            f"    model = joblib.load('{model_file}')",
            "except Exception as exc:",
            "    model = None",
            "    model_error = str(exc)",
            "",
            "@app.get('/health')",
            "def health():",
            "    if model is None:",
            "        return {'status': 'unhealthy', 'error': model_error}",
            "    return {'status': 'healthy'}",
            "",
            "def validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:",
            "    if not isinstance(payload, dict):",
            "        raise ValueError('Payload must be a dictionary.')",
            "    required = [",
            fields,
            "    ]",
            "    missing = [name for name in required if name not in payload]",
            "    if missing:",
            "        raise ValueError(f'Missing features: {missing}')",
            "    return {name: payload[name] for name in required}",
            "",
            "@app.post('/predict')",
            "def predict(request: PredictRequest):",
            "    if model is None:",
            "        raise HTTPException(status_code=500, detail='Model failed to load.')",
            "    try:",
            "        features = validate_payload(request.features)",
            "        df = pd.DataFrame([features])",
            "        prediction = model.predict(df)",
            "        confidence = None",
            "        if hasattr(model, 'predict_proba'):",
            "            probabilities = model.predict_proba(df)",
            "            confidence = float(max(probabilities[0]))",
            "        return {'prediction': prediction[0], 'confidence': confidence}",
            "    except ValueError as exc:",
            "        raise HTTPException(status_code=422, detail=str(exc))",
            "    except Exception as exc:",
            "        raise HTTPException(status_code=500, detail=str(exc))",
        ])

    def generate_requirements(self) -> List[str]:
        """Return recommended requirements for generated code."""
        return self.DEFAULT_REQUIREMENTS.copy()

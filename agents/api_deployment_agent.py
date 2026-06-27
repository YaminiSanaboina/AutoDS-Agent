import datetime
import json
import os
import pickle
from typing import Any, Dict, List, Optional


class APIDeploymentAgent:
    DEFAULT_DEPLOYMENT_DIR = "deployment_package"
    DEFAULT_MODEL_FILE = "model.pkl"
    DEFAULT_APP_FILE = "main.py"
    DEFAULT_REQUIREMENTS = [
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=1.10.0",
        "scikit-learn>=1.4.0",
        "pandas>=2.0.0",
        "numpy>=1.25.0",
    ]
    DEFAULT_DOCKERFILE = "Dockerfile"
    DEFAULT_README = "README.md"
    DEFAULT_REGISTRY_FILE = "deployment_history.json"

    def __init__(self, deployment_dir: Optional[str] = None, history_path: Optional[str] = None) -> None:
        self.deployment_dir = deployment_dir or self.DEFAULT_DEPLOYMENT_DIR
        self.history_path = history_path or self.DEFAULT_REGISTRY_FILE
        self._ensure_directory(self.deployment_dir)
        self.history = self._load_history()

    def _ensure_directory(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def _load_history(self) -> Dict[str, Any]:
        if not os.path.exists(self.history_path):
            return {"deployments": []}
        try:
            with open(self.history_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and "deployments" in data:
                return data
        except (OSError, json.JSONDecodeError):
            pass
        return {"deployments": []}

    def _persist_history(self) -> None:
        with open(self.history_path, "w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)

    def generate_fastapi_service(
        self,
        model_file: str,
        feature_names: List[str],
        output_file: Optional[str] = None,
        title: str = "AutoDS Model API",
    ) -> Dict[str, Any]:
        output_file = output_file or os.path.join(self.deployment_dir, self.DEFAULT_APP_FILE)
        self._ensure_directory(os.path.dirname(output_file) or self.deployment_dir)

        validation_code = self._build_feature_validation(feature_names)
        service_code = self._build_fastapi_code(feature_names, validation_code, title)

        with open(output_file, "w", encoding="utf-8") as handle:
            handle.write(service_code)

        return {
            "status": "success",
            "app_file": output_file,
            "model_file": model_file,
        }

    def build_deployment_package(
        self,
        model_source_path: str,
        feature_names: List[str],
        deployment_name: Optional[str] = None,
        requirements: Optional[List[str]] = None,
        dockerfile_name: Optional[str] = None,
        readme_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._ensure_directory(self.deployment_dir)
        package_name = deployment_name or self.deployment_dir
        app_path = os.path.join(self.deployment_dir, self.DEFAULT_APP_FILE)
        requirements_path = os.path.join(self.deployment_dir, requirements or "requirements.txt")
        dockerfile_path = os.path.join(self.deployment_dir, dockerfile_name or self.DEFAULT_DOCKERFILE)
        readme_path = os.path.join(self.deployment_dir, readme_name or self.DEFAULT_README)
        model_target_path = os.path.join(self.deployment_dir, os.path.basename(model_source_path))

        if not os.path.exists(model_source_path):
            return {"status": "failed", "reason": "Model source file does not exist."}

        with open(model_source_path, "rb") as src, open(model_target_path, "wb") as dst:
            dst.write(src.read())

        service_result = self.generate_fastapi_service(model_target_path, feature_names, app_path)
        requirements_list = requirements or self.DEFAULT_REQUIREMENTS

        with open(requirements_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(requirements_list) + "\n")

        dockerfile_text = self._build_dockerfile(os.path.basename(self.DEFAULT_APP_FILE), os.path.basename(requirements_path))
        with open(dockerfile_path, "w", encoding="utf-8") as handle:
            handle.write(dockerfile_text)

        readme_text = self._build_readme_text(package_name, feature_names, os.path.basename(model_target_path))
        with open(readme_path, "w", encoding="utf-8") as handle:
            handle.write(readme_text)

        return {
            "status": "success",
            "app_file": app_path,
            "requirements_file": requirements_path,
            "dockerfile": dockerfile_path,
            "readme": readme_path,
            "model_file": model_target_path,
        }

    def _build_feature_validation(self, feature_names: List[str]) -> str:
        if not feature_names:
            return "    return payload['features']"

        lines = ["    features = payload.get('features', {})"]
        lines.append("    if not isinstance(features, dict):")
        lines.append("        raise ValueError('features must be a dictionary')")
        lines.append("    missing = [f for f in [")
        for feature in feature_names:
            lines.append(f"        '{feature}',")
        lines.append("    ] if f not in features]")
        lines.append("    if missing:")
        lines.append("        raise ValueError(f'Missing input features: {missing}')")
        lines.append("    return [features[f] for f in [")
        for feature in feature_names:
            lines.append(f"        '{feature}',")
        lines.append("    ]")
        return "\n".join(lines)

    def _build_fastapi_code(self, feature_names: List[str], validation_code: str, title: str) -> str:
        return f"""from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
from typing import Any, Dict


class PredictionRequest(BaseModel):
    features: Dict[str, Any]


app = FastAPI(title=\"{title}\")

MODEL_PATH = os.path.join(os.path.dirname(__file__), \"{os.path.basename(self.DEFAULT_MODEL_FILE)}\")

try:
    with open(MODEL_PATH, \"rb\") as model_handle:
        model = pickle.load(model_handle)
except Exception as exc:
    model = None
    model_load_error = str(exc)


@app.get('/health')
def health() -> Dict[str, str]:
    if model is None:
        return {{"status": "unhealthy", "error": model_load_error}}
    return {{"status": "healthy"}}


@app.post('/predict')
def predict(request: PredictionRequest) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=500, detail='Model failed to load')

    payload = request.dict()
{validation_code}
    try:
        prediction = model.predict([features])
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([features])
            confidence = float(max(proba[0]))
        elif hasattr(model, 'decision_function'):
            score = model.decision_function([features])
            confidence = float(abs(score[0]))
        return {{"prediction": prediction[0], "confidence": confidence}}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
"""

    def _build_dockerfile(self, app_file: str, requirements_file: str) -> str:
        return (
            "FROM python:3.12-slim\n"
            "WORKDIR /app\n"
            "COPY . /app\n"
            "RUN python -m pip install --no-cache-dir -r " + requirements_file + "\n"
            "EXPOSE 8000\n"
            "CMD [\"uvicorn\", \"{app_file}:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n"
        )

    def _build_readme_text(self, package_name: str, feature_names: List[str], model_filename: str) -> str:
        return (
            f"# {package_name}\n"
            "\n"
            "This package contains a FastAPI prediction service wrapping a trained AutoDS model.\n"
            "\n"
            "## Contents\n"
            f"- {model_filename}\n"
            "- main.py\n"
            "- requirements.txt\n"
            "- Dockerfile\n"
            "\n"
            "## Run locally\n"
            "```bash\n"
            "python -m pip install -r requirements.txt\n"
            "uvicorn main:app --host 0.0.0.0 --port 8000\n"
            "```\n"
            "\n"
            "## API Endpoints\n"
            "- GET /health\n"
            "- POST /predict\n"
            "\n"
            "## Input sample\n"
            "```json\n"
            "{\n"
            "  \"features\": {\n"
            + ",\n".join([f'  \"{name}\": 0' for name in feature_names])
            + "\n}\n"
            "}\n"
            "```\n"
        )

    def calculate_deployment_readiness(
        self,
        model_quality: float,
        data_quality: float,
        explainability_available: bool,
        tests_completed: bool,
        drift_monitoring_configured: bool,
    ) -> Dict[str, Any]:
        score = 0.0
        score += min(100.0, max(0.0, model_quality)) * 0.4
        score += min(100.0, max(0.0, data_quality)) * 0.25
        score += (100.0 if explainability_available else 0.0) * 0.15
        score += (100.0 if tests_completed else 0.0) * 0.1
        score += (100.0 if drift_monitoring_configured else 0.0) * 0.1

        final_score = float(round(min(100.0, max(0.0, score)), 1))
        if final_score >= 90:
            status = "Production Ready"
        elif final_score >= 70:
            status = "Needs Minor Improvements"
        else:
            status = "Not Ready for Deployment"

        return {
            "score": final_score,
            "status": status,
            "components": {
                "model_quality": float(round(min(100.0, max(0.0, model_quality)), 1)),
                "data_quality": float(round(min(100.0, max(0.0, data_quality)), 1)),
                "explainability": 100.0 if explainability_available else 0.0,
                "testing": 100.0 if tests_completed else 0.0,
                "drift_monitoring": 100.0 if drift_monitoring_configured else 0.0,
            },
        }

    def record_deployment_event(
        self,
        model_id: str,
        version: str,
        environment: str,
        api_url: str,
        status: str = "Deployed",
    ) -> Dict[str, Any]:
        deployment_id = f"DEPLOY_{len(self.history.get('deployments', [])) + 1:03d}"
        entry = {
            "deployment_id": deployment_id,
            "model_id": model_id,
            "version": version,
            "environment": environment,
            "deployment_date": datetime.datetime.utcnow().isoformat() + "Z",
            "status": status,
            "api_url": api_url,
        }
        self.history.setdefault("deployments", []).append(entry)
        self._persist_history()
        return entry

    def get_deployment_history(self) -> List[Dict[str, Any]]:
        return list(self.history.get("deployments", []))

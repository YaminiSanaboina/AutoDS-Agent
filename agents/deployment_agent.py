"""Deployment and MLOps agent for AutoDS Agent.

This backend-only module prepares trained models for production deployment,
packages them with metadata, generates API and container artifacts, and
provides monitoring and risk guidance.
"""
from __future__ import annotations

import json
import os
import pickle
import datetime
from typing import Any, Dict, List, Optional, Tuple

from agents.experiment_memory_agent import ExperimentMemoryAgent
from agents.hyperparameter_agent import HyperparameterOptimizationAgent
from agents.self_healing_agent import SelfHealingAgent
from agents.supervisor_agent import AISupervisor


class DeploymentAgent:
    DEFAULT_PACKAGE_DIR = "model_package"
    REGISTRY_FILE = "storage/registry/model_registry.json"
    DEFAULT_REQUIREMENTS = [
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "scikit-learn>=1.4.0",
        "pandas>=2.0.0",
        "numpy>=1.25.0",
    ]

    def __init__(self, package_dir: Optional[str] = None, registry_path: Optional[str] = None) -> None:
        self.package_dir = package_dir or self.DEFAULT_PACKAGE_DIR
        self.registry_path = registry_path or self.REGISTRY_FILE
        self.memory_agent = ExperimentMemoryAgent()
        self.self_healing_agent = SelfHealingAgent()
        self.hyperparameter_agent = HyperparameterOptimizationAgent()
        self.supervisor = self._load_supervisor()

    def _load_supervisor(self) -> Optional[AISupervisor]:
        try:
            return AISupervisor()
        except Exception:
            return None

    def _ensure_directory(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def _normalize_model_key(self, model_name: str) -> str:
        return model_name.strip().replace(" ", "_").upper()

    def _load_registry(self) -> Dict[str, Any]:
        if not os.path.exists(self.registry_path):
            return {}
        try:
            with open(self.registry_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            return {}

    def _persist_registry(self, registry: Dict[str, Any]) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as handle:
            json.dump(registry, handle, indent=2)

    def package_model(
        self,
        model: Any,
        model_name: str,
        dataset_name: str,
        problem_type: str,
        feature_list: List[str],
        metrics: Dict[str, float],
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_info: Optional[Dict[str, Any]] = None,
        intended_usage: Optional[str] = None,
        limitations: Optional[str] = None,
        ethical_considerations: Optional[str] = None,
        package_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Package a trained model for production deployment."""
        target_dir = package_path or self.package_dir
        self._ensure_directory(target_dir)

        package_output: Dict[str, Any] = {
            "package_path": target_dir,
            "files": {},
            "status": "failed",
        }

        try:
            model_file = os.path.join(target_dir, "model.pkl")
            metadata_file = os.path.join(target_dir, "metadata.json")
            requirements_file = os.path.join(target_dir, "requirements.txt")
            readme_file = os.path.join(target_dir, "README.md")

            with open(model_file, "wb") as handle:
                pickle.dump(model, handle)

            metadata = {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "problem_type": problem_type,
                "feature_list": feature_list,
                "metrics": metrics,
                "hyperparameters": hyperparameters or {},
                "training_info": training_info or {},
                "intended_usage": intended_usage or "Batch and online prediction for production use.",
                "limitations": limitations or "Model should be validated on fresh data before deployment.",
                "ethical_considerations": ethical_considerations or "Use responsibly and monitor for bias or drift.",
                "packaged_at": datetime.datetime.utcnow().isoformat() + "Z",
            }

            version_entry = self.create_version(
                model_name=model_name,
                changes=training_info.get("changes", "Initial production deployment.") if training_info else "Initial production deployment.",
                performance_improvement=metrics.get("performance_improvement") if metrics else None,
                package_path=target_dir,
            )
            metadata["version"] = version_entry["version"]

            with open(metadata_file, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)

            with open(requirements_file, "w", encoding="utf-8") as handle:
                handle.write("\n".join(self.DEFAULT_REQUIREMENTS) + "\n")

            readme_text = self._build_readme_text(metadata)
            with open(readme_file, "w", encoding="utf-8") as handle:
                handle.write(readme_text)

            self._record_deployment_event(metadata, "packaged")

            package_output["files"] = {
                "model": model_file,
                "metadata": metadata_file,
                "requirements": requirements_file,
                "readme": readme_file,
            }
            package_output["metadata"] = metadata
            package_output["status"] = "success"
            return package_output
        except Exception as error:
            diagnostics = str(error)
            recovery = self.self_healing_agent.recommend_fix(
                self.self_healing_agent.analyze_error(diagnostics)
            )
            return {
                "status": "failed",
                "diagnostics": diagnostics,
                "recovery": recovery,
            }

    def _build_readme_text(self, metadata: Dict[str, Any]) -> str:
        return (
            f"# Model Package: {metadata.get('model_name')}\n"
            f"\n"
            f"## Version\n"
            f"{metadata.get('version')}\n"
            f"\n"
            f"## Dataset\n"
            f"{metadata.get('dataset_name')}\n"
            f"\n"
            f"## Problem Type\n"
            f"{metadata.get('problem_type')}\n"
            f"\n"
            f"## Features\n"
            f"{', '.join(metadata.get('feature_list', []))}\n"
            f"\n"
            f"## Metrics\n"
            f"{json.dumps(metadata.get('metrics', {}), indent=2)}\n"
            f"\n"
            f"## Usage\n"
            f"{metadata.get('intended_usage')}\n"
            f"\n"
            f"## Limitations\n"
            f"{metadata.get('limitations')}\n"
            f"\n"
            f"## Ethical Considerations\n"
            f"{metadata.get('ethical_considerations')}\n"
            f"\n"
            f"## API\n"
            f"Run `uvicorn app:app --host 0.0.0.0 --port 8000` after installing dependencies.\n"
        )

    def generate_api(
        self,
        package_path: Optional[str] = None,
        app_file_name: str = "app.py",
        requirements_file_name: str = "requirements.txt",
    ) -> Dict[str, Any]:
        """Generate FastAPI deployment code for the packaged model."""
        target_dir = package_path or self.package_dir
        self._ensure_directory(target_dir)

        api_file = os.path.join(target_dir, app_file_name)
        requirements_file = os.path.join(target_dir, requirements_file_name)
        metadata_file = os.path.join(target_dir, "metadata.json")

        try:
            if not os.path.exists(metadata_file):
                raise FileNotFoundError("metadata.json is required to generate API code.")

            with open(metadata_file, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)

            features = metadata.get("feature_list", [])
            feature_validation = self._build_feature_validation(features)
            app_code = self._build_api_code(features, feature_validation)

            with open(api_file, "w", encoding="utf-8") as handle:
                handle.write(app_code)

            api_requirements = list(self.DEFAULT_REQUIREMENTS) + ["fastapi>=0.104.0", "uvicorn[standard]>=0.24.0"]
            with open(requirements_file, "w", encoding="utf-8") as handle:
                handle.write("\n".join(sorted(set(api_requirements))) + "\n")

            return {
                "status": "success",
                "app_file": api_file,
                "requirements_file": requirements_file,
            }
        except Exception as error:
            diagnostics = str(error)
            recovery = self.self_healing_agent.recommend_fix(
                self.self_healing_agent.analyze_error(diagnostics)
            )
            return {
                "status": "failed",
                "diagnostics": diagnostics,
                "recovery": recovery,
            }

    def _build_feature_validation(self, features: List[str]) -> str:
        if not features:
            return "    return [value for value in payload.values()]"

        lines = [
            "    required_features = [",
            *[f"        \"{feature}\"," for feature in features],
            "    ]",
            "    missing = [f for f in required_features if f not in payload]",
            "    if missing:",
            "        raise ValueError(f'Missing features: {missing}')",
            "    data_row = []",
            "    for feature in required_features:",
            "        if payload[feature] is None:",
            "            raise ValueError(f'Feature {feature} cannot be null.')",
            "        data_row.append(payload[feature])",
            "    return data_row",
        ]
        return "\n".join(lines)

    def _build_api_code(self, features: List[str], feature_validation: str) -> str:
        features_block = ", ".join([f'\"{name}\"' for name in features])
        if not features_block:
            features_block = ""

        return f"""import pickle
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title=\"AutoDS Deployment API\")

MODEL_FILE = \"model.pkl\"

class PredictionRequest(BaseModel):
    payload: Dict[str, Any]


def load_model():
    with open(MODEL_FILE, \"rb\") as handle:
        return pickle.load(handle)

MODEL = load_model()


def validate_payload(payload: Dict[str, Any]) -> List[Any]:
{feature_validation}

@app.get(\"/\")
def health_check():
    return {{\"status\": \"ok\", \"message\": \"Model deployment API is ready.\"}}

@app.post(\"/predict\")
def predict(request: PredictionRequest):
    payload = request.payload
    try:
        data_row = validate_payload(payload)
        features = np.array([data_row])
        prediction = MODEL.predict(features).tolist()
        confidence = None
        if hasattr(MODEL, \"predict_proba\"):
            probabilities = MODEL.predict_proba(features)
            confidence = float(np.max(probabilities))
        elif hasattr(MODEL, \"decision_function\"):
            scores = MODEL.decision_function(features)
            confidence = float(np.max(scores))

        return {{
            \"prediction\": prediction,
            \"confidence\": confidence,
            \"features\": data_row,
        }}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
"""

    def generate_docker_files(
        self,
        package_path: Optional[str] = None,
        dockerfile_name: str = "Dockerfile",
        compose_name: str = "docker-compose.yml",
    ) -> Dict[str, Any]:
        """Generate Docker deployment artifacts."""
        target_dir = package_path or self.package_dir
        self._ensure_directory(target_dir)

        dockerfile_path = os.path.join(target_dir, dockerfile_name)
        compose_path = os.path.join(target_dir, compose_name)

        try:
            dockerfile_contents = (
                "FROM python:3.12-slim\n"
                "WORKDIR /app\n"
                "COPY requirements.txt /app/requirements.txt\n"
                "RUN python -m pip install --no-cache-dir -r requirements.txt\n"
                "COPY . /app\n"
                "EXPOSE 8000\n"
                "CMD [\"uvicorn\", \"app:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n"
            )

            compose_contents = (
                "version: '3.8'\n"
                "services:\n"
                "  api:\n"
                "    build: .\n"
                "    ports:\n"
                "      - \"8000:8000\"\n"
                "    volumes:\n"
                "      - .:/app\n"
                "    command: uvicorn app:app --host 0.0.0.0 --port 8000\n"
            )

            with open(dockerfile_path, "w", encoding="utf-8") as handle:
                handle.write(dockerfile_contents)

            with open(compose_path, "w", encoding="utf-8") as handle:
                handle.write(compose_contents)

            return {
                "status": "success",
                "dockerfile": dockerfile_path,
                "docker_compose": compose_path,
            }
        except Exception as error:
            diagnostics = str(error)
            recovery = self.self_healing_agent.recommend_fix(
                self.self_healing_agent.analyze_error(diagnostics)
            )
            return {
                "status": "failed",
                "diagnostics": diagnostics,
                "recovery": recovery,
            }

    def generate_model_card(
        self,
        model_name: str,
        dataset_name: str,
        problem_type: str,
        training_date: Optional[str],
        metrics: Dict[str, float],
        features: List[str],
        intended_usage: Optional[str] = None,
        limitations: Optional[str] = None,
        ethical_considerations: Optional[str] = None,
        version: Optional[str] = None,
    ) -> str:
        """Return a Markdown model card describing deployment details."""
        training_date = training_date or datetime.datetime.utcnow().isoformat() + "Z"
        intended_usage = intended_usage or "Use for predictive inference in production environments."
        limitations = limitations or "Monitor the model for drift and retrain when performance degrades."
        ethical_considerations = ethical_considerations or "Ensure fairness and avoid using sensitive attributes without review."

        metrics_text = "\n".join([f"- **{key}**: {value}" for key, value in metrics.items()])
        features_text = "\n".join([f"- {feature}" for feature in features])

        return (
            f"# Model Card: {model_name}\n\n"
            f"**Version:** {version or 'TBD'}\n\n"
            f"## Dataset\n"
            f"{dataset_name}\n\n"
            f"## Problem Type\n"
            f"{problem_type}\n\n"
            f"## Training Date\n"
            f"{training_date}\n\n"
            f"## Performance Metrics\n"
            f"{metrics_text}\n\n"
            f"## Features Used\n"
            f"{features_text}\n\n"
            f"## Intended Usage\n"
            f"{intended_usage}\n\n"
            f"## Limitations\n"
            f"{limitations}\n\n"
            f"## Ethical Considerations\n"
            f"{ethical_considerations}\n\n"
            f"## Monitoring\n"
            f"Retrain if accuracy drops more than 5% or if data drift is detected.\n"
        )

    def create_version(
        self,
        model_name: str,
        changes: str,
        performance_improvement: Optional[float] = None,
        package_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create or update model version metadata in the registry."""
        registry = self._load_registry()
        model_key = self._normalize_model_key(model_name)
        versions = registry.get(model_key, [])
        next_version = self._next_version(versions, performance_improvement)

        entry = {
            "version": next_version,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "changes": changes,
            "performance_improvement": performance_improvement,
        }
        versions.append(entry)
        registry[model_key] = versions
        self._persist_registry(registry)

        if package_path:
            copy_path = os.path.join(package_path, os.path.basename(self.registry_path))
            try:
                with open(copy_path, "w", encoding="utf-8") as handle:
                    json.dump(registry, handle, indent=2)
            except OSError:
                pass

        return entry

    def _normalize_dict_like(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if value is None:
            return {}
        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                normalized = value.to_dict()
                if isinstance(normalized, dict):
                    return normalized
            except Exception:
                pass
        return {}

    def _normalize_scalar(self, value: Any) -> Any:
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return value[0]
        return value

    def _next_version(self, existing_versions: List[Dict[str, Any]], performance_improvement: Optional[float]) -> str:
        if not existing_versions:
            return "MODEL_v1.0"

        last_version = existing_versions[-1].get("version", "MODEL_v1.0")
        prefix, _, version_suffix = last_version.rpartition("_v")
        if not version_suffix:
            prefix = last_version
            version_suffix = "1.0"

        try:
            major, minor = [int(part) for part in version_suffix.split(".")]
        except ValueError:
            major, minor = 1, 0

        if performance_improvement is not None and performance_improvement >= 0.05:
            major += 1
            minor = 0
        else:
            minor += 1

        return f"{prefix}_v{major}.{minor}"

    def analyze_deployment_risk(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze deployment risk from packaged model metadata."""
        metrics = self._normalize_dict_like(metadata.get("metrics", {}))
        training_info = self._normalize_dict_like(metadata.get("training_info", {}))
        feature_list = metadata.get("feature_list", [])
        hyperparameters = self._normalize_dict_like(metadata.get("hyperparameters", {}))

        warnings: List[str] = []
        risk_score = 0

        accuracy = self._normalize_scalar(metrics.get("accuracy") or metrics.get("test_score"))
        train_score = self._normalize_scalar(metrics.get("train_score"))
        test_score = self._normalize_scalar(metrics.get("test_score"))

        if metrics is None or (isinstance(metrics, dict) and len(metrics) == 0):
            warnings.append("Model metrics are missing.")
            risk_score += 3
        else:
            if accuracy is None and test_score is None:
                warnings.append("No baseline accuracy or test score is available.")
                risk_score += 2
            elif accuracy is not None:
                try:
                    if float(accuracy) < 0.7:
                        warnings.append("Model accuracy is lower than expected for production.")
                        risk_score += 2
                except (TypeError, ValueError):
                    pass

        if train_score is not None and test_score is not None:
            try:
                gap = float(train_score) - float(test_score)
                if gap >= 0.15:
                    warnings.append("Overfitting gap is large; model may not generalize.")
                    risk_score += 3
                elif gap >= 0.10:
                    warnings.append("Moderate overfitting gap detected.")
                    risk_score += 2
            except (TypeError, ValueError):
                pass

        if not feature_list:
            warnings.append("Feature list is missing from metadata.")
            risk_score += 2

        preprocessing = training_info.get("preprocessing_steps")
        if preprocessing is None or (hasattr(preprocessing, "__len__") and len(preprocessing) == 0):
            warnings.append("No preprocessing steps were recorded.")
            risk_score += 1

        explainability = training_info.get("explainability_available")
        if explainability is None or explainability is False:
            warnings.append("Explainability analysis has not been documented.")
            risk_score += 1

        if hyperparameters is None or (isinstance(hyperparameters, dict) and len(hyperparameters) == 0):
            warnings.append("Hyperparameter details are missing.")
            risk_score += 1

        if risk_score >= 6:
            level = "High"
        elif risk_score >= 3:
            level = "Medium"
        else:
            level = "Low"

        recommendations = []
        if "Model metrics are missing." in warnings:
            recommendations.append("Record evaluation metrics such as accuracy, R², or F1-score.")
        if "Overfitting gap is large; model may not generalize." in warnings:
            recommendations.append("Add regularization, simplify the model, or gather more data.")
        if "Feature list is missing from metadata." in warnings:
            recommendations.append("Document the production feature set precisely.")
        if "No preprocessing steps were recorded." in warnings:
            recommendations.append("Capture the preprocessing pipeline used during training.")
        if "Explainability analysis has not been documented." in warnings:
            recommendations.append("Run explainability analysis and attach SHAP or feature importance results.")
        if "Hyperparameter details are missing." in warnings:
            recommendations.append("Store optimizer parameters for repeatable deployment." )
        if not recommendations:
            recommendations.append("Deploy with monitoring and schedule retraining if performance drifts.")

        return {
            "risk_level": level,
            "warnings": warnings,
            "recommendations": recommendations,
        }

    def generate_monitoring_plan(self, model_name: str, dataset_name: str) -> Dict[str, Any]:
        """Generate a monitoring blueprint for the deployed model."""
        return {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "plan": [
                "Log every prediction request and response with timestamps.",
                "Check data drift weekly by comparing new input distributions to training data.",
                "Monitor model performance with accuracy, F1-score, or R² depending on problem type.",
                "Trigger alerts if performance drops more than 5% from baseline.",
                "Retrain when data drift is detected or performance degrades consistently.",
                "Review deploy logs for errors and invoke recovery guidance from the SelfHealingAgent.",
            ],
            "retrains": [
                "Retrain if accuracy drops more than 5%.",
                "Retrain if the feature input distribution shifts significantly.",
                "Retrain when new labeled data becomes available.",
            ],
        }

    def _record_deployment_event(self, metadata: Dict[str, Any], status: str) -> None:
        self.memory_agent.log_experiment(
            dataset_name=metadata.get("dataset_name", "Unknown Dataset"),
            problem_type=metadata.get("problem_type", "Unknown"),
            algorithm_name=metadata.get("model_name", "Unknown Model"),
            hyperparameters=metadata.get("hyperparameters", {}),
            train_score=metadata.get("metrics", {}).get("train_score"),
            test_score=metadata.get("metrics", {}).get("test_score"),
            cv_score=metadata.get("metrics", {}).get("cv_score"),
            training_time=metadata.get("training_info", {}).get("training_time"),
            feature_count=len(metadata.get("feature_list", [])),
            notes=f"Deployment event: {status}",
        )

        if self.supervisor:
            try:
                self.supervisor.add_decision_log("deployment_event", {"status": status, "model_name": metadata.get("model_name")})
            except Exception:
                # Supervisor logging should not block deployment packaging
                pass

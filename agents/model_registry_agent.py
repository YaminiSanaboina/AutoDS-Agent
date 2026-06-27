import datetime
import json
import os
from typing import Any, Dict, List, Optional, Tuple


class ModelRegistryAgent:
    DEFAULT_REGISTRY_PATH = "storage/registry/model_registry.json"

    def __init__(self, registry_path: Optional[str] = None) -> None:
        self.registry_path = registry_path or self.DEFAULT_REGISTRY_PATH
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        if not os.path.exists(self.registry_path):
            return {"models": []}
        try:
            with open(self.registry_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and "models" in data:
                return data
        except (OSError, json.JSONDecodeError):
            pass
        return {"models": []}

    def _persist_registry(self) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as handle:
            json.dump(self.registry, handle, indent=2)

    def _next_model_id(self) -> str:
        model_count = len(self.registry.get("models", [])) + 1
        return f"MODEL_{model_count:03d}"

    def _next_version(self, model_name: str) -> str:
        versions = [entry.get("version") for entry in self.registry.get("models", []) if entry.get("model_name") == model_name]
        next_index = len(versions) + 1
        return f"MODEL_v{next_index}"

    def register_model(
        self,
        model: Any,
        model_name: str,
        dataset_name: str,
        problem_type: str,
        algorithm: str,
        metrics: Dict[str, float],
        feature_names: List[str],
        hyperparameters: Dict[str, Any],
        training_time: float,
        shap_available: bool = False,
        deployment_status: str = "Not Deployed",
        artifact_path: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        model_id = self._next_model_id()
        version = self._next_version(model_name)
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"

        entry: Dict[str, Any] = {
            "model_id": model_id,
            "timestamp": timestamp,
            "version": version,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "problem_type": problem_type,
            "algorithm": algorithm,
            "feature_names": feature_names,
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "training_time": training_time,
            "shap_available": shap_available,
            "deployment_status": deployment_status,
            "artifact_path": artifact_path,
            "is_active": True,
            "extra_metadata": extra_metadata or {},
        }

        for existing in self.registry.get("models", []):
            if existing.get("model_name") == model_name:
                existing["is_active"] = False

        self.registry.setdefault("models", []).append(entry)
        self._persist_registry()
        return entry

    def get_model_versions(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        models = self.registry.get("models", [])
        if model_name is None:
            return sorted(models, key=lambda item: item.get("timestamp", ""))
        return sorted(
            [item for item in models if item.get("model_name") == model_name],
            key=lambda item: item.get("timestamp", ""),
        )

    def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        return next((item for item in self.registry.get("models", []) if item.get("model_id") == model_id), None)

    def compare_versions(self, model_id_a: str, model_id_b: str) -> Dict[str, Any]:
        a = self.get_model_by_id(model_id_a)
        b = self.get_model_by_id(model_id_b)
        if not a or not b:
            return {"error": "One or both model IDs were not found."}

        metrics_a = a.get("metrics", {})
        metrics_b = b.get("metrics", {})
        keys = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))
        comparison: List[Dict[str, Any]] = []
        a_score = 0.0
        b_score = 0.0
        count = 0

        for key in keys:
            value_a = metrics_a.get(key)
            value_b = metrics_b.get(key)
            if value_a is None or value_b is None:
                comparison.append({"metric": key, "a": value_a, "b": value_b, "difference": None})
                continue
            diff = float(value_b) - float(value_a)
            comparison.append({"metric": key, "a": value_a, "b": value_b, "difference": diff})
            a_score += float(value_a)
            b_score += float(value_b)
            count += 1

        recommendation = "Both versions have similar performance. Review metrics before choosing."
        if count > 0:
            if b_score > a_score:
                recommendation = f"Upgrade to {b['version']} ({b['algorithm']}) based on higher aggregate performance."
            elif a_score > b_score:
                recommendation = f"Keep {a['version']} ({a['algorithm']}) as the stronger version."

        return {
            "model_a": {"model_id": a["model_id"], "version": a["version"], "algorithm": a["algorithm"]},
            "model_b": {"model_id": b["model_id"], "version": b["version"], "algorithm": b["algorithm"]},
            "comparison": comparison,
            "recommendation": recommendation,
        }

    def rollback_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        target = self.get_model_by_id(model_id)
        if not target:
            return None

        model_name = target.get("model_name")
        for entry in self.registry.get("models", []):
            if entry.get("model_name") == model_name:
                entry["is_active"] = False
        target["is_active"] = True
        target["deployment_status"] = "Rolled Back"
        self._persist_registry()
        return target

    def _normalize_value(self, value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if numeric > 1.0:
            numeric = numeric / 100.0
        return max(0.0, min(1.0, numeric))

    def _score_model(self, metrics: Dict[str, Any], problem_type: str) -> float:
        if problem_type == "Classification":
            keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        else:
            keys = ["r2", "mae", "rmse"]

        total = 0.0
        count = 0
        for key in keys:
            if key in metrics:
                value = self._normalize_value(metrics[key])
                if key in {"mae", "rmse"}:
                    value = 1.0 - value
                total += value
                count += 1

        return float(round((total / count) * 100.0, 1)) if count else 0.0

    def generate_leaderboard(self, problem_type: str) -> List[Dict[str, Any]]:
        matched = [entry for entry in self.registry.get("models", []) if entry.get("problem_type") == problem_type]
        ranked: List[Tuple[float, Dict[str, Any]]] = []
        for entry in matched:
            score = self._score_model(entry.get("metrics", {}), problem_type)
            ranked.append((score, entry))
        ranked.sort(key=lambda item: item[0], reverse=True)

        leaderboard = []
        for rank, (score, entry) in enumerate(ranked, start=1):
            leaderboard.append(
                {
                    "rank": rank,
                    "model_name": entry.get("model_name"),
                    "version": entry.get("version"),
                    "dataset": entry.get("dataset_name"),
                    "score": score,
                    "status": entry.get("deployment_status", "Not Deployed"),
                }
            )
        return leaderboard

from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict, Iterable, List, Optional

from utils.safe_checks import feature_importance_as_dict, is_present, normalize_recommendations, safe_dict_get


class DocumentationAgent:
    """Agent for generating documentation and managing a knowledge base."""

    def __init__(self, knowledge_path: str = "storage/memory/knowledge_base.json") -> None:
        self.knowledge_path = knowledge_path
        self._ensure_knowledge_file()

    def _ensure_knowledge_file(self) -> None:
        if not os.path.exists(self.knowledge_path):
            with open(self.knowledge_path, "w", encoding="utf-8") as handle:
                json.dump([], handle, indent=2)

    def _load_knowledge_base(self) -> List[Dict[str, Any]]:
        try:
            with open(self.knowledge_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_knowledge_base(self, entries: List[Dict[str, Any]]) -> None:
        with open(self.knowledge_path, "w", encoding="utf-8") as handle:
            json.dump(entries[:10000], handle, indent=2, ensure_ascii=False)

    def generate_project_documentation(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        project_data = self._normalize_project_data(project_data)
        title = project_data.get("project_title") or project_data.get("title") or project_data.get("dataset_name") or "AutoDS Analysis Project"
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dataset_name = project_data.get("dataset_name", "Uploaded Dataset")

        sections = {
            "Business Objective": project_data.get(
                "business_objective",
                project_data.get("project_goal", "Deliver predictive insights from the uploaded dataset."),
            ),
            "Dataset Summary": project_data.get(
                "dataset_summary",
                f"The dataset '{dataset_name}' was analyzed for structure, quality, and predictive potential.",
            ),
            "Data Preprocessing": project_data.get(
                "preprocessing_steps",
                "Automated cleaning, missing value handling, duplicate review, and encoding were applied where needed.",
            ),
            "Feature Engineering": project_data.get(
                "feature_engineering_decisions",
                "Candidate transformations and feature suggestions were generated to improve model performance.",
            ),
            "Model Selection": project_data.get(
                "model_selection_rationale",
                "Multiple models were trained and the best performer was selected using validation metrics.",
            ),
            "Hyperparameter Optimization": project_data.get(
                "hyperparameter_summary",
                "Hyperparameter recommendations were generated for the selected algorithm family.",
            ),
            "Evaluation Metrics": project_data.get(
                "evaluation_metrics",
                "Models were compared using problem-appropriate validation scores.",
            ),
            "Explainability Summary": project_data.get(
                "explainability_summary",
                "Feature importance and SHAP-style explanations highlight the strongest model drivers.",
            ),
            "Deployment Recommendation": project_data.get(
                "deployment_recommendation",
                "Deploy after reviewing readiness score, risk level, and monitoring plan.",
            ),
            "Future Improvements": project_data.get(
                "future_improvements",
                "Collect more data, monitor drift, and iterate on feature engineering.",
            ),
        }

        summary = project_data.get(
            "summary",
            (
                f"{title} analyzed {dataset_name} end-to-end: data quality review, EDA, model training, "
                f"explainability, and deployment readiness. Best model: {project_data.get('best_model', 'N/A')}."
            ),
        )

        return {
            "title": title,
            "created_at": created_at,
            "sections": sections,
            "summary": summary,
        }

    def _normalize_project_data(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map pipeline-style payloads into documentation-friendly fields."""
        if not isinstance(project_data, dict):
            return {}

        normalized = dict(project_data)
        dataset_report = normalized.get("dataset_report") or normalized.get("dataset_analysis") or {}
        if isinstance(dataset_report, dict):
            normalized.setdefault("dataset_name", dataset_report.get("dataset_name"))
            problem = dataset_report.get("problem_analysis") or {}
            normalized.setdefault(
                "dataset_summary",
                problem.get("summary") or dataset_report.get("summary"),
            )
            normalized.setdefault("problem_type", problem.get("problem_type"))

        cleaning = normalized.get("cleaning_results") or {}
        if isinstance(cleaning, dict):
            report = cleaning.get("report") or {}
            actions = report.get("actions") if isinstance(report, dict) else None
            if is_present(actions):
                normalized.setdefault("preprocessing_steps", "; ".join(str(a) for a in actions[:8]))

        feature_plan = normalized.get("feature_plan") or normalized.get("feature_engineering_results") or {}
        if isinstance(feature_plan, dict):
            steps = feature_plan.get("steps") or feature_plan.get("recommended_changes") or []
            if is_present(steps):
                normalized.setdefault(
                    "feature_engineering_decisions",
                    "; ".join(str(s) for s in steps[:8]),
                )

        model_results = normalized.get("model_results") or normalized.get("strategy") or {}
        if isinstance(model_results, dict):
            normalized.setdefault("best_model", model_results.get("best_model"))
            metrics = model_results.get("metrics") or {}
            if metrics:
                top = sorted(metrics.items(), key=lambda x: x[1], reverse=True)[:5]
                normalized.setdefault(
                    "evaluation_metrics",
                    ", ".join(f"{name}: {score:.4f}" for name, score in top),
                )
                normalized.setdefault(
                    "model_selection_rationale",
                    f"{model_results.get('best_model', 'Top model')} achieved the highest validation score among evaluated candidates.",
                )

        hyperparameter_report = normalized.get("hyperparameter_report") or {}
        if isinstance(hyperparameter_report, dict):
            params = hyperparameter_report.get("recommended_parameters") or hyperparameter_report.get("parameters") or {}
            if params:
                normalized.setdefault(
                    "hyperparameter_summary",
                    ", ".join(f"{k}={v}" for k, v in list(params.items())[:6]),
                )

        explainability = normalized.get("explainability_results") or normalized.get("xai_results") or {}
        if isinstance(explainability, dict):
            fi = feature_importance_as_dict(safe_dict_get(explainability, "feature_importance"))
            if fi:
                top_feats = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                normalized.setdefault(
                    "explainability_summary",
                    "Top drivers: " + ", ".join(f"{k} ({v:.3f})" for k, v in top_feats),
                )

        deployment = normalized.get("deployment_readiness") or {}
        if isinstance(deployment, dict):
            risk = deployment.get("risk_level", "Unknown")
            recs = normalize_recommendations(deployment.get("recommendations"))
            normalized.setdefault(
                "deployment_recommendation",
                f"Risk level: {risk}. " + ("; ".join(str(r) for r in recs[:3]) if recs else "Review monitoring and API packaging before production."),
            )

        normalized.setdefault("project_title", normalized.get("project_goal") or normalized.get("dataset_name"))
        return normalized

    def generate_dataset_card(self, dataset_report: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "dataset_name": dataset_report.get("dataset_name", "Unnamed Dataset"),
            "source": dataset_report.get("source", "Unknown source"),
            "samples": int(dataset_report.get("samples", dataset_report.get("sample_count", 0))),
            "features": dataset_report.get("features", []),
            "target_variable": dataset_report.get("target_variable", dataset_report.get("target", None)),
            "domain": dataset_report.get("domain", "General"),
            "quality_score": int(dataset_report.get("quality_score", 0)),
            "risks": dataset_report.get("risks", []),
            "recommended_usage": dataset_report.get(
                "recommended_usage",
                "Use for exploratory analysis, model training, and validation while monitoring data quality.",
            ),
        }

    def generate_model_card(
        self,
        model_info: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        ethics_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        model_name = model_info.get("model_name", "Unnamed Model")
        training_config = model_info.get("training_configuration", {})
        strengths = model_info.get(
            "strengths",
            ["Good fit for the problem type.", "Balanced performance and interpretability."],
        )
        limitations = model_info.get(
            "limitations", ["May require retraining as data drifts."])

        fairness_notes = ethics_report.get("fairness_considerations", []) if ethics_report else model_info.get(
            "fairness_considerations", ["Fairness evaluation should be completed before deployment."])
        responsible_ai = ethics_report.get("responsible_ai_notes", []) if ethics_report else model_info.get(
            "responsible_ai_notes", ["Review bias, privacy, and transparency for responsible deployment."])

        return {
            "model_details": {
                "name": model_name,
                "framework": model_info.get("framework", "Unknown"),
                "version": model_info.get("version", "TBD"),
                "description": model_info.get("description", "Detailed model information is available."),
            },
            "training_configuration": training_config,
            "metrics": performance_metrics,
            "strengths": strengths,
            "limitations": limitations,
            "fairness_considerations": fairness_notes,
            "responsible_ai_notes": responsible_ai,
            "deployment_readiness": model_info.get(
                "deployment_readiness",
                "Assess performance, fairness, and monitoring readiness before production deployment.",
            ),
        }

    def generate_experiment_report(self, experiment_history: Dict[str, Any]) -> Dict[str, Any]:
        experiments = experiment_history.get("experiments", [])
        total_experiments = len(experiments)
        best_model = None
        best_score = None
        for experiment in experiments:
            score = experiment.get("score")
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_model = experiment

        failed_experiments = [exp for exp in experiments if not exp.get("success", True)]
        improvements = experiment_history.get("performance_improvements", [])
        lessons = experiment_history.get("lessons_learned", [])

        return {
            "total_experiments": total_experiments,
            "best_model": best_model or {},
            "performance_improvements": improvements,
            "failed_experiments": failed_experiments,
            "lessons_learned": lessons,
        }

    def generate_api_documentation(self, api_metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "api_name": api_metadata.get("api_name", "Unnamed API"),
            "description": api_metadata.get(
                "description",
                "Describes endpoints, request formats, responses, error codes, and authentication requirements.",
            ),
            "endpoints": api_metadata.get("endpoints", []),
            "authentication": api_metadata.get("authentication", "Token-based authentication is recommended."),
            "error_codes": api_metadata.get("error_codes", []),
            "examples": api_metadata.get("examples", []),
        }

    def save_knowledge_entry(
        self,
        category: str,
        title: str,
        content: str,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        entries = self._load_knowledge_base()
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "category": category,
            "title": title,
            "content": content,
            "tags": tags or [],
        }
        entries.insert(0, entry)
        self._save_knowledge_base(entries)
        return entry

    def search_knowledge(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        entries = self._load_knowledge_base()
        query_lower = query.lower()
        results: List[Dict[str, Any]] = []
        for entry in entries:
            if category and entry.get("category") != category:
                continue
            if tags and not set(tags).issubset(set(entry.get("tags", []))):
                continue
            text_search = " ".join(
                [entry.get("title", ""), entry.get("content", ""), " ".join(entry.get("tags", []))]
            ).lower()
            if query_lower in text_search:
                results.append(entry)
        return results

    def list_entries(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._load_knowledge_base()[:limit]

    def export_markdown(self, document: Dict[str, Any]) -> str:
        lines: List[str] = []
        title = document.get("title")
        if title:
            lines.append(f"# {title}")

        summary = document.get("summary")
        if is_present(summary):
            lines.append(summary)

        sections = document.get("sections", {})
        for heading, content in sections.items():
            lines.append(f"## {heading}")
            if isinstance(content, dict):
                for key, value in content.items():
                    lines.append(f"- **{key}**: {value}")
            else:
                lines.append(str(content))

        if not lines:
            return ""
        return "\n\n".join(lines)

    def export_json(self, document: Dict[str, Any]) -> str:
        return json.dumps(document, ensure_ascii=False, indent=2, default=str)

    def export_text(self, document: Dict[str, Any]) -> str:
        lines: List[str] = []
        title = document.get("title")
        if title:
            lines.append(title)
        summary = document.get("summary")
        if is_present(summary):
            lines.append(summary)
        sections = document.get("sections", {})
        for heading, content in sections.items():
            lines.append(f"{heading}:")
            if isinstance(content, dict):
                for key, value in content.items():
                    lines.append(f"  {key}: {value}")
            else:
                lines.append(f"  {content}")
        return "\n".join(lines)

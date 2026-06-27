"""Multi-Agent Collaboration Engine for AutoDS Agent.

This backend-only module enables AI agents to communicate, debate, vote,
resolve conflicts, and execute autonomous project workflows without modifying
Streamlit UI pages.
"""
from __future__ import annotations

import datetime
import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from agents.chief_data_scientist_agent import ChiefDataScientistAgent
from agents.dataset_intelligence_agent import DatasetIntelligenceAgent
from agents.deployment_agent import DeploymentAgent
from agents.experiment_memory_agent import ExperimentMemoryAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.hyperparameter_agent import HyperparameterOptimizationAgent
from agents.self_healing_agent import SelfHealingAgent
from agents.drift_monitoring_agent import DataDriftMonitoringAgent
from agents.retraining_agent import AutonomousRetrainingAgent
from agents.supervisor_agent import AISupervisor
from agents.workflow_agent import AIWorkflowAgent


class MultiAgentCollaborationEngine:
    DEFAULT_HISTORY_FILE = "storage/monitoring/agent_collaboration_history.json"
    MAX_HISTORY = 1000

    AGENT_WEIGHTS = {
        "Chief Data Scientist": 5,
        "Supervisor": 3,
        "Feature Engineering": 2,
        "Model": 2,
        "Drift": 2,
        "Deployment": 2,
    }

    def __init__(
        self,
        history_path: Optional[str] = None,
        drift_reference_path: Optional[str] = None,
        drift_history_path: Optional[str] = None,
    ) -> None:
        self.history_path = history_path or self.DEFAULT_HISTORY_FILE
        self.history = self._load_history()

        self.chief = ChiefDataScientistAgent()
        self.supervisor = self._safe_init(AISupervisor)
        self.workflow_agent = self._safe_init(AIWorkflowAgent)
        self.dataset_agent: Optional[DatasetIntelligenceAgent] = None
        self.feature_agent: Optional[FeatureEngineeringAgent] = None
        self.hyperparameter_agent = HyperparameterOptimizationAgent()
        self.experiment_memory_agent = ExperimentMemoryAgent()
        self.self_healing_agent = SelfHealingAgent()
        self.drift_agent = DataDriftMonitoringAgent(
            reference_path=drift_reference_path,
            history_path=drift_history_path,
        )
        self.retraining_agent = AutonomousRetrainingAgent()
        self.deployment_agent = DeploymentAgent()

    def _safe_init(self, cls):
        try:
            return cls()
        except Exception:
            return None

    def _load_history(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.history_path):
            return []
        try:
            with open(self.history_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
        return []

    def _persist_history(self) -> None:
        directory = os.path.dirname(self.history_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(self.history_path, "w", encoding="utf-8") as handle:
            json.dump(self.history[-self.MAX_HISTORY :], handle, indent=2)

    def _record_meeting(
        self,
        participants: List[str],
        opinions: List[Dict[str, Any]],
        debate: Dict[str, Any],
        final_decision: str,
        confidence: float,
    ) -> None:
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "participants": participants,
            "opinions": opinions,
            "debate": debate,
            "final_decision": final_decision,
            "confidence": float(round(confidence, 2)),
        }
        self.history.append(entry)
        self.history = self.history[-self.MAX_HISTORY :]
        self._persist_history()

    def _extract_decision_label(self, opinion: str) -> str:
        normalized = opinion.lower()
        if "retrain" in normalized or "retraining" in normalized or "drift" in normalized:
            return "retrain"
        if "deploy" in normalized or "deployment" in normalized or "production" in normalized:
            return "deploy"
        if "clean" in normalized or "impute" in normalized or "missing" in normalized or "data quality" in normalized:
            return "clean_data"
        if "feature" in normalized or "engineer" in normalized or "strategy" in normalized:
            return "feature_engineering"
        if "monitor" in normalized or "wait" in normalized or "continue" in normalized:
            return "monitor"
        if "collect more data" in normalized or "gather more data" in normalized or "insufficient data" in normalized:
            return "collect_data"
        if "explain" in normalized or "interpret" in normalized or "trustworthiness" in normalized:
            return "explainability"
        return "undecided"

    def _normalize_confidence(self, confidence: float) -> float:
        if confidence is None:
            return 0.5
        return min(1.0, max(0.0, float(confidence)))

    def start_agent_meeting(
        self,
        project_state: Optional[Dict[str, Any]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        model_info: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        project_state = project_state or {}
        dataset_info = dataset_info or {}
        model_info = model_info or {}
        opinions: List[Dict[str, Any]] = []

        dataset = self._extract_dataframe(dataset_info)
        if dataset is not None:
            self.dataset_agent = DatasetIntelligenceAgent(dataset, dataset_info.get("name", "Untitled Dataset"))
            self.feature_agent = FeatureEngineeringAgent(dataset, dataset_info.get("name", "Untitled Dataset"))

        opinions.extend(self._collect_opinions_from_agents(project_state, dataset_info, model_info))
        self._record_meeting(
            participants=[item["agent"] for item in opinions],
            opinions=opinions,
            debate={"arguments": [], "counterarguments": [], "confidence_scores": {}},
            final_decision="pending",
            confidence=0.0,
        )
        return opinions

    def _extract_dataframe(self, dataset_info: Dict[str, Any]) -> Optional[pd.DataFrame]:
        if isinstance(dataset_info, pd.DataFrame):
            return dataset_info
        if isinstance(dataset_info.get("dataframe"), pd.DataFrame):
            return dataset_info.get("dataframe")
        if isinstance(dataset_info.get("df"), pd.DataFrame):
            return dataset_info.get("df")
        return None

    def _safe_call(self, func, default: Any = None, *args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception:
            return default

    def _collect_opinions_from_agents(
        self,
        project_state: Dict[str, Any],
        dataset_info: Dict[str, Any],
        model_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        opinions: List[Dict[str, Any]] = []

        # Chief Data Scientist
        chief_opinion = self._generate_chief_opinion(project_state, dataset_info, model_info)
        opinions.append(chief_opinion)

        # Supervisor
        opinions.append(self._generate_supervisor_opinion(project_state, dataset_info, model_info))

        # Workflow
        opinions.append(self._generate_workflow_opinion(project_state, dataset_info, model_info))

        # Dataset Intelligence
        opinions.append(self._generate_dataset_opinion(project_state, dataset_info, model_info))

        # Feature Engineering
        opinions.append(self._generate_feature_opinion(project_state, dataset_info, model_info))

        # Model and Hyperparameter
        opinions.append(self._generate_model_opinion(project_state, dataset_info, model_info))

        # Experiment Memory
        opinions.append(self._generate_experiment_memory_opinion(project_state, dataset_info, model_info))

        # Self Healing
        opinions.append(self._generate_self_healing_opinion(project_state, dataset_info, model_info))

        # Drift Monitoring
        opinions.append(self._generate_drift_opinion(project_state, dataset_info, model_info))

        # Autonomous Retraining
        opinions.append(self._generate_retraining_opinion(project_state, dataset_info, model_info))

        # Deployment
        opinions.append(self._generate_deployment_opinion(project_state, dataset_info, model_info))

        return opinions

    def _generate_chief_opinion(self, project_state, dataset_info, model_info) -> Dict[str, Any]:
        recommendations = self._safe_call(
            lambda: self.chief.give_strategic_advice(
                dataset_report=dataset_info.get("dataset_report"),
                model_history=model_info.get("history"),
                drift_history=project_state.get("drift_history"),
                deployment_status=project_state.get("deployment_status"),
                risk_assessment=dataset_info.get("risk_assessment"),
            ),
            default=[{"recommendation": "Review the project and decide the best next step.", "priority": "Medium"}],
        )
        choice = recommendations[0] if recommendations else {"recommendation": "Proceed with a careful review."}
        opinion = choice.get("recommendation", "Proceed with a careful review.")
        return {
            "agent": "Chief Data Scientist",
            "opinion": opinion,
            "confidence": self._normalize_confidence(float(choice.get("confidence", 0.9) if isinstance(choice, dict) else 0.9)),
            "reasoning": "Aggregates dataset, model, drift, and deployment signals into a high-level recommendation.",
        }

    def _generate_supervisor_opinion(self, project_state, dataset_info, model_info) -> Dict[str, Any]:
        if self.supervisor is not None:
            supervisor_result = self._safe_call(self.supervisor.recommend_next_agent, default={
                "next_agent": "AISupervisor",
                "reason": "Evaluate project workflow and delegate the next step.",
                "recommendations": ["Review the current stage and proceed."]
            })
            opinion = supervisor_result.get("reason", "Evaluate the workflow and delegate the next task.")
        else:
            opinion = "Supervisor is unavailable and recommends a workflow review before taking the next step."
        return {
            "agent": "Supervisor",
            "opinion": opinion,
            "confidence": 0.82,
            "reasoning": "Monitors workflow state and recommends the next agent to act.",
        }

    def _generate_workflow_opinion(self, project_state, dataset_info, model_info) -> Dict[str, Any]:
        if self.workflow_agent is not None:
            workflow_action = self._safe_call(self.workflow_agent.recommend_next_action, default="Assess the dataset and take the next available workflow step.")
            opinion = workflow_action if isinstance(workflow_action, str) else str(workflow_action)
        else:
            opinion = "Workflow agent is unavailable; continue with dataset evaluation and monitoring."
        return {
            "agent": "Workflow",
            "opinion": opinion,
            "confidence": 0.75,
            "reasoning": "Provides a concise next action based on workflow state and progress.",
        }

    def _generate_dataset_opinion(self, project_state, dataset_info, model_info) -> Dict[str, Any]:
        if self.dataset_agent is not None:
            report = self._safe_call(self.dataset_agent.generate_dataset_report, default={})
            problem = report.get("problem_analysis", {}).get("problem_type") if isinstance(report, dict) else None
            risk = report.get("risk_analysis", {}).get("risk_level") if isinstance(report, dict) else None
            opinion = f"Dataset requires careful preparation. Problem type: {problem or 'unknown'}. Risk level: {risk or 'unknown'}."
            confidence = 0.78
        else:
            opinion = "Dataset intelligence is unavailable; inspect the dataset for quality and target selection.",
            confidence = 0.6
        return {
            "agent": "Dataset Intelligence",
            "opinion": opinion,
            "confidence": confidence,
            "reasoning": "Analyzes dataset quality, problem type, and risk before modeling.",
        }

    def _generate_feature_opinion(self, project_state, dataset_info, model_info) -> Dict[str, Any]:
        if self.feature_agent is not None:
            feature_plan = self._safe_call(self.feature_agent.generate_feature_plan, default={})
            steps = feature_plan.get("steps") if isinstance(feature_plan, dict) else None
            opinion = "Additional feature engineering is recommended." if steps else "Feature strategy is stable."
            confidence = 0.8
            reasoning = "Feature engineering can improve model signal and address quality issues."
        else:
            opinion = "Feature engineering agent is unavailable; review feature strategy manually."
            confidence = 0.6
            reasoning = "The feature agent normally recommends encoding, imputation, and derived features."
        return {
            "agent": "Feature Engineering",
            "opinion": opinion,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    def _generate_model_opinion(self, project_state, dataset_info, model_info) -> Dict[str, Any]:
        problem_type = model_info.get("problem_type") or "Classification"
        supported = self._safe_call(lambda: self.hyperparameter_agent.supported_models(problem_type), default=[])
        if supported:
            opinion = f"Evaluate candidate models such as {supported[:3]}." if isinstance(supported, list) else "Evaluate candidate models for the task."
            confidence = 0.77
        else:
            opinion = "Model selection is unclear; validate the problem type and available estimators."
            confidence = 0.6
        return {
            "agent": "Model",
            "opinion": opinion,
            "confidence": confidence,
            "reasoning": "Recommends model selection and tuning based on the inferred problem type.",
        }

    def _generate_experiment_memory_opinion(self, project_state, dataset_info, model_info) -> Dict[str, Any]:
        memory = getattr(self.experiment_memory_agent, "memory", {})
        experiment_count = len(memory.get("experiments", []))
        if experiment_count:
            opinion = f"Use lessons from {experiment_count} past experiments to refine the next model."
            confidence = 0.7
        else:
            opinion = "No prior experiments are available; use standard AutoML best practices."
            confidence = 0.55
        return {
            "agent": "Experiment Memory",
            "opinion": opinion,
            "confidence": confidence,
            "reasoning": "Leverages past experiments and memory to avoid repeating failed approaches.",
        }

    def _generate_self_healing_opinion(self, project_state, dataset_info, model_info) -> Dict[str, Any]:
        health_report = self._safe_call(self.self_healing_agent.generate_health_report, default={})
        if health_report.get("total_failures", 0) > 0:
            opinion = "Review recent failure patterns and apply safe recovery steps before pushing to production."
            confidence = 0.85
        else:
            opinion = "No known AutoML failure history found; proceed with the standard lifecycle."
            confidence = 0.65
        return {
            "agent": "Self Healing",
            "opinion": opinion,
            "confidence": confidence,
            "reasoning": "Monitors error history and recommends safe recovery actions when needed.",
        }

    def _generate_drift_opinion(self, project_state, dataset_info, model_info) -> Dict[str, Any]:
        drift_report = project_state.get("drift_report") or {}
        severity = drift_report.get("severity") if isinstance(drift_report, dict) else None
        if severity == "High":
            opinion = "Do not deploy yet because production drift risk is high. Retraining should be prioritized."
            confidence = 0.9
        elif severity == "Medium":
            opinion = "Monitor drift closely and prepare a retraining plan if performance drops."
            confidence = 0.8
        else:
            opinion = "Drift risk is low. Favor deployment if model performance is strong."
            confidence = 0.75
        return {
            "agent": "Drift Monitoring",
            "opinion": opinion,
            "confidence": confidence,
            "reasoning": "Assesses data drift risk and its impact on deployment timing.",
        }

    def _generate_retraining_opinion(self, project_state, dataset_info, model_info) -> Dict[str, Any]:
        drift_report = project_state.get("drift_report") or {}
        performance = model_info.get("performance", {})
        retrain_decision = self._safe_call(
            lambda: self.retraining_agent.should_retrain(drift_report, current_model_performance=performance),
            default={"decision": False, "reason": "Evaluate retraining conditions before action.", "priority": "Low"},
        )
        if retrain_decision.get("decision"):
            opinion = f"Retrain the model because: {retrain_decision.get('reason')}."
            confidence = 0.88
        else:
            opinion = f"Do not retrain for now. {retrain_decision.get('reason')}"
            confidence = 0.65
        return {
            "agent": "Retraining",
            "opinion": opinion,
            "confidence": confidence,
            "reasoning": "Evaluates whether drift or performance degradation require retraining.",
        }

    def _generate_deployment_opinion(self, project_state, dataset_info, model_info) -> Dict[str, Any]:
        deployment_status = project_state.get("deployment_status") or {}
        if deployment_status.get("deployed"):
            opinion = "The model is already deployed; continue monitoring performance and drift."
            confidence = 0.8
        else:
            opinion = "Package and deploy the best model once explainability and validation checks pass."
            confidence = 0.78
        return {
            "agent": "Deployment",
            "opinion": opinion,
            "confidence": confidence,
            "reasoning": "Advises on deployment readiness and packaging requirements.",
        }

    def run_agent_debate(
        self,
        opinions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if opinions is None:
            opinions = self.start_agent_meeting()

        arguments = []
        counterarguments = []
        counts: Dict[str, int] = {}

        for opinion in opinions:
            label = self._extract_decision_label(opinion["opinion"])
            arguments.append({
                "agent": opinion["agent"],
                "argument": opinion["opinion"],
                "decision": label,
                "confidence": opinion["confidence"],
            })
            counts[label] = counts.get(label, 0) + 1

        if len(set(item["decision"] for item in arguments)) > 1:
            for i, arg in enumerate(arguments):
                for other in arguments[i + 1 :]:
                    if arg["decision"] != other["decision"]:
                        counterarguments.append(
                            {
                                "agent": other["agent"],
                                "counterargument": f"While {arg['agent']} recommends {arg['decision']}, {other['agent']} prefers {other['decision']}.",
                                "confidence": other["confidence"],
                            }
                        )

        confidence_scores = {
            decision: float(round(sum(item["confidence"] for item in arguments if item["decision"] == decision) / max(1, sum(1 for item in arguments if item["decision"] == decision)), 2))
            for decision in counts
        }

        return {
            "arguments": arguments,
            "counterarguments": counterarguments,
            "confidence_scores": confidence_scores,
        }

    def generate_consensus(
        self,
        opinions: Optional[List[Dict[str, Any]]] = None,
        project_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        opinions = opinions or self.start_agent_meeting(project_state=project_state)
        project_state = project_state or {}

        votes: Dict[str, float] = {}
        total_weights = 0.0

        for opinion in opinions:
            label = self._extract_decision_label(opinion["opinion"])
            weight = self.AGENT_WEIGHTS.get(opinion["agent"], 1)
            votes[label] = votes.get(label, 0.0) + weight * opinion["confidence"]
            total_weights += weight

        if not votes:
            final_decision = "monitor"
            confidence = 0.5
        else:
            final_decision = max(votes, key=votes.get)
            confidence = float(round(min(1.0, votes[final_decision] / max(1.0, total_weights)), 2))

        if self._has_conflict(opinions):
            resolution = self.resolve_agent_conflicts(opinions, project_state)
            if resolution and resolution.get("final_decision"):
                final_decision = resolution["final_decision"]
                confidence = max(confidence, float(round(resolution.get("confidence", confidence), 2)))

        return {
            "final_decision": final_decision,
            "confidence": confidence,
            "votes": {k: float(round(v, 2)) for k, v in votes.items()},
        }

    def _has_conflict(self, opinions: List[Dict[str, Any]]) -> bool:
        return len({self._extract_decision_label(item["opinion"]) for item in opinions}) > 1

    def resolve_agent_conflicts(
        self,
        opinions: Optional[List[Dict[str, Any]]] = None,
        project_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        opinions = opinions or self.start_agent_meeting(project_state=project_state)
        project_state = project_state or {}

        drift_severity = (project_state.get("drift_report") or {}).get("severity", "Low")
        health_score = float(project_state.get("health_score", 50.0))
        risk_level = (project_state.get("risk_level") or "Medium").lower()
        domain = (project_state.get("domain") or "").lower()

        if drift_severity == "High":
            return {
                "final_decision": "retrain",
                "confidence": 0.95,
                "reason": "High drift risk outweighs deployment recommendations.",
            }

        if health_score < 50.0 or risk_level == "high":
            if any(self._extract_decision_label(op["opinion"]) == "clean_data" for op in opinions):
                return {
                    "final_decision": "clean_data",
                    "confidence": 0.9,
                    "reason": "Poor project health or risk level indicates data quality should be resolved first.",
                }
            return {
                "final_decision": "monitor",
                "confidence": 0.7,
                "reason": "Defer production actions until project health improves.",
            }

        if domain == "healthcare" and any(self._extract_decision_label(op["opinion"]) == "deploy" for op in opinions):
            if any("explain" in op["opinion"].lower() or "trust" in op["opinion"].lower() for op in opinions):
                return {
                    "final_decision": "explainability",
                    "confidence": 0.88,
                    "reason": "Healthcare domain requires interpretable decisions and trust before deployment.",
                }

        if any(self._extract_decision_label(op["opinion"]) == "retrain" for op in opinions) and any(self._extract_decision_label(op["opinion"]) == "deploy" for op in opinions):
            return {
                "final_decision": "retrain",
                "confidence": 0.9,
                "reason": "Retraining is prioritized when model performance and drift are in tension.",
            }

        if any("insufficient data" in op["opinion"].lower() or "collect more data" in op["opinion"].lower() for op in opinions):
            return {
                "final_decision": "collect_data",
                "confidence": 0.86,
                "reason": "The team agrees that data collection is the safer next step than deployment.",
            }

        return {
            "final_decision": self.generate_consensus(opinions, project_state).get("final_decision", "monitor"),
            "confidence": self.generate_consensus(opinions, project_state).get("confidence", 0.6),
            "reason": "Consensus was used because no strong conflict resolution rule applied.",
        }

    def generate_team_report(
        self,
        meeting_record: Optional[Dict[str, Any]] = None,
        project_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meeting_record = meeting_record or (self.history[-1] if self.history else {})
        project_state = project_state or {}
        participants = meeting_record.get("participants", [])
        final_decision = meeting_record.get("final_decision", "pending")
        confidence = meeting_record.get("confidence", 0.0)
        risks = project_state.get("risk_level") or project_state.get("drift_report", {}).get("severity") or "Unknown"
        next_actions: List[str] = []

        for opinion in meeting_record.get("opinions", []):
            recommendation = opinion.get("opinion", "")
            if recommendation and len(next_actions) < 3:
                next_actions.append(recommendation)

        return {
            "team_members": participants,
            "key_discussions": [op["opinion"] for op in meeting_record.get("opinions", [])],
            "final_decision": final_decision,
            "confidence": confidence,
            "risks": risks,
            "recommended_next_actions": next_actions,
            "summary": f"The AI collaboration team met and reached a decision of '{final_decision}' with confidence {confidence}.",
        }

    def execute_autonomous_project(
        self,
        dataset: pd.DataFrame,
        dataset_metadata: Optional[Dict[str, Any]] = None,
        model_info: Optional[Dict[str, Any]] = None,
        project_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        dataset_metadata = dataset_metadata or {}
        model_info = model_info or {}
        project_state = project_state or {}
        timeline: List[Dict[str, Any]] = []
        agents_involved: List[str] = []

        timeline.append({
            "step": "Dataset arrival",
            "agent": "Dataset Intelligence",
            "description": f"Dataset '{dataset_metadata.get('name', 'Untitled Dataset')}' arrived with {dataset.shape[0]} rows and {dataset.shape[1]} columns.",
        })
        agents_involved.append("Dataset Intelligence")

        analysis = self.chief.analyze_project(
            dataset=dataset,
            dataset_metadata=dataset_metadata,
            model_history=model_info.get("history", []),
            drift_history=project_state.get("drift_history", []),
            deployment_status=project_state.get("deployment_status", {}),
        )
        timeline.append({
            "step": "Chief analysis",
            "agent": "Chief Data Scientist",
            "description": analysis.get("next_best_action", "Produce strategic guidance."),
        })
        agents_involved.append("Chief Data Scientist")

        supervisor_action = self._safe_call(self.supervisor.recommend_next_agent, default={
            "next_agent": "DataAgent",
            "reason": "Fallback to dataset preparation.",
        }) if self.supervisor else {"next_agent": "DataAgent", "reason": "Supervisor unavailable."}
        timeline.append({
            "step": "Supervisor assignment",
            "agent": "Supervisor",
            "description": supervisor_action.get("reason", "Assign the next task in the workflow."),
        })
        agents_involved.append("Supervisor")

        self.dataset_agent = DatasetIntelligenceAgent(dataset, dataset_metadata.get("name", "Untitled Dataset"))
        dataset_report = self._safe_call(self.dataset_agent.generate_dataset_report, default={})
        timeline.append({
            "step": "Dataset evaluation",
            "agent": "Dataset Intelligence",
            "description": dataset_report.get("executive_summary", "Analyze dataset quality and risks."),
        })

        if self.feature_agent is None:
            self.feature_agent = FeatureEngineeringAgent(dataset, dataset_metadata.get("name", "Untitled Dataset"))
        feature_plan = self._safe_call(self.feature_agent.generate_feature_plan, default={})
        timeline.append({
            "step": "Feature strategy",
            "agent": "Feature Engineering",
            "description": feature_plan.get("expected_benefit", "Generate a feature engineering plan."),
        })
        agents_involved.append("Feature Engineering")

        model_recommendation = self._safe_call(
            lambda: self.hyperparameter_agent.supported_models(model_info.get("problem_type", "Classification")),
            default=["Logistic Regression"],
        )
        timeline.append({
            "step": "Model selection",
            "agent": "Model",
            "description": f"Selected candidate models: {model_recommendation[:3]}.",
        })
        agents_involved.append("Model")

        xai_description = "Evaluate explainability and trustworthiness using SHAP-like analysis."
        timeline.append({
            "step": "XAI validation",
            "agent": "Self Healing",
            "description": xai_description,
        })
        agents_involved.append("Self Healing")

        deploy_description = "Prepare deployment packaging after validation checks are complete."
        timeline.append({
            "step": "Deployment preparation",
            "agent": "Deployment",
            "description": deploy_description,
        })
        agents_involved.append("Deployment")

        self.drift_agent.register_reference_data(dataset, dataset_name=dataset_metadata.get("name", "reference_dataset"))
        timeline.append({
            "step": "Drift monitoring",
            "agent": "Drift Monitoring",
            "description": "Reference data has been registered and drift monitoring begins.",
        })
        agents_involved.append("Drift Monitoring")

        meeting_opinions = self.start_agent_meeting(
            project_state=project_state,
            dataset_info={"dataframe": dataset, "name": dataset_metadata.get("name", "Untitled Dataset"), "dataset_report": dataset_report},
            model_info=model_info,
        )
        consensus = self.generate_consensus(meeting_opinions, project_state=project_state)
        debate = self.run_agent_debate(meeting_opinions)
        meeting_record = {
            "participants": [opinion["agent"] for opinion in meeting_opinions],
            "opinions": meeting_opinions,
            "debate": debate,
            "final_decision": consensus["final_decision"],
            "confidence": consensus["confidence"],
        }
        self._record_meeting(
            participants=meeting_record["participants"],
            opinions=meeting_record["opinions"],
            debate=meeting_record["debate"],
            final_decision=meeting_record["final_decision"],
            confidence=meeting_record["confidence"],
        )

        timeline.append({
            "step": "Final consensus",
            "agent": "Multi-Agent Collaboration Engine",
            "description": f"Final decision: {consensus['final_decision']} with confidence {consensus['confidence']}.",
        })

        return {
            "timeline": timeline,
            "final_decision": consensus["final_decision"],
            "confidence": consensus["confidence"],
            "agents_involved": list(dict.fromkeys(agents_involved)),
            "meeting_record": meeting_record,
        }

    def get_collaboration_history(self) -> List[Dict[str, Any]]:
        return list(self.history)

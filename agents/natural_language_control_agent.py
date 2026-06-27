import json
import os
import datetime
from typing import Any, Dict, List, Optional, Tuple

from agents.autonomous_data_scientist_agent import AutonomousDataScientistAgent
from agents.dataset_intelligence_agent import DatasetIntelligenceAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.hyperparameter_agent import HyperparameterOptimizationAgent
from agents.model_registry_agent import ModelRegistryAgent
from agents.api_deployment_agent import APIDeploymentAgent
from agents.drift_monitoring_agent import DataDriftMonitoringAgent
from agents.code_generation_agent import CodeGenerationAgent
from agents.notebook_generation_agent import NotebookGenerationAgent
from agents.chief_data_scientist_agent import ChiefDataScientistAgent


class NaturalLanguageControlAgent:
    DEFAULT_MEMORY_FILE = "ai_control_memory.json"
    MAX_MEMORY = 5000

    INTENT_MAP = {
        "analyze_data": ["analyze my data", "check data quality", "find problems in dataset"],
        "train_model": ["train the best model", "improve accuracy", "try better algorithms"],
        "optimize_hyperparameters": ["optimize hyperparameters", "tune hyperparameters"],
        "explainability": ["explain why the model predicted this", "show important features"],
        "deploy_model": ["is my model ready for production", "deploy my model"],
        "check_drift": ["check data drift", "is there drift"],
        "generate_report": ["generate business report", "export notebook", "generate python code"],
        "autonomous_mode": ["do everything automatically", "act as my ai data scientist", "be my ai data scientist"],
    }

    ROUTE_AGENTS = {
        "analyze_data": "DatasetIntelligenceAgent",
        "train_model": "HyperparameterOptimizationAgent",
        "optimize_hyperparameters": "HyperparameterOptimizationAgent",
        "explainability": "XAIAgent",
        "deploy_model": "APIDeploymentAgent",
        "check_drift": "DataDriftMonitoringAgent",
        "generate_report": "NotebookGenerationAgent",
        "autonomous_mode": "AutonomousDataScientistAgent",
    }

    def __init__(self, memory_path: Optional[str] = None) -> None:
        self.memory_path = memory_path or self.DEFAULT_MEMORY_FILE
        self.memory: List[Dict[str, Any]] = self._load_memory()
        # instantiate lightweight controllers (not executed until routed)
        self.autonomous = AutonomousDataScientistAgent()
        self.dataset_agent_cls = DatasetIntelligenceAgent
        self.feature_agent_cls = FeatureEngineeringAgent
        self.hyper_agent = HyperparameterOptimizationAgent()
        self.registry_agent = ModelRegistryAgent()
        self.deploy_agent = APIDeploymentAgent()
        self.drift_agent = DataDriftMonitoringAgent()
        self.code_agent = CodeGenerationAgent()
        self.notebook_agent = NotebookGenerationAgent()
        self.chief_agent = ChiefDataScientistAgent()

    def _load_memory(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.memory_path):
            return []
        try:
            with open(self.memory_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
        except (OSError, json.JSONDecodeError):
            pass
        return []

    def _persist_memory(self) -> None:
        directory = os.path.dirname(self.memory_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as fh:
            json.dump(self.memory[-self.MAX_MEMORY :], fh, indent=2)

    def record_interaction(self, user_message: str, intent: str, actions: List[Dict[str, Any]], results: Dict[str, Any]) -> None:
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "user_message": user_message,
            "intent": intent,
            "actions": actions,
            "results": results,
        }
        self.memory.append(entry)
        self.memory = self.memory[-self.MAX_MEMORY :]
        self._persist_memory()

    def understand_command(self, user_message: str) -> Dict[str, Any]:
        msg = user_message.lower()
        for intent, phrases in self.INTENT_MAP.items():
            for p in phrases:
                if p in msg:
                    # extract entities (simple rule-based)
                    entities = {}
                    if "accuracy" in msg or "improve" in msg:
                        entities["goal"] = "maximize accuracy"
                    if "interpret" in msg or "explain" in msg:
                        entities["goal"] = "improve interpretability"
                    return {"intent": intent, "confidence": 0.95, "entities": entities}

        # Fallback: classify keywords
        if "deploy" in msg or "production" in msg:
            return {"intent": "deploy_model", "confidence": 0.8, "entities": {}}
        if "notebook" in msg or "export" in msg or "code" in msg:
            return {"intent": "generate_report", "confidence": 0.8, "entities": {}}

        return {"intent": "unknown", "confidence": 0.5, "entities": {}}

    def route_command(self, intent: str) -> str:
        return self.ROUTE_AGENTS.get(intent, "ChiefDataScientistAgent")

    def create_execution_plan(self, intent: str, entities: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        entities = entities or {}
        plan: List[Dict[str, Any]] = []
        if intent in {"train_model", "optimize_hyperparameters"}:
            plan = [
                {"step": 1, "action": "Review previous experiments"},
                {"step": 2, "action": "Analyze feature quality"},
                {"step": 3, "action": "Run hyperparameter optimization"},
                {"step": 4, "action": "Compare new models"},
                {"step": 5, "action": "Register improved model"},
            ]
        elif intent == "analyze_data":
            plan = [
                {"step": 1, "action": "Run Dataset Intelligence analysis"},
                {"step": 2, "action": "Produce risk assessment"},
                {"step": 3, "action": "Recommend feature engineering steps"},
            ]
        elif intent == "autonomous_mode":
            plan = [
                {"step": 1, "action": "Run full autonomous project"},
                {"step": 2, "action": "Persist decisions"},
                {"step": 3, "action": "Produce executive report"},
            ]
        else:
            plan = [{"step": 1, "action": f"Route to {self.route_command(intent)}"}]
        return plan

    def safety_check(self, intent: str, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        context = context or {}
        # Simple safety rules
        if intent in {"train_model", "optimize_hyperparameters"}:
            if not context.get("dataset_loaded"):
                return False, "Cannot optimize models because no dataset is loaded. Load a dataset first."
        if intent == "deploy_model":
            if not context.get("model_available"):
                return False, "Cannot deploy because no trained model is available. Train a model first."
        return True, "OK"

    def act_on_command(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = context or {}
        understood = self.understand_command(user_message)
        intent = understood.get("intent")
        entities = understood.get("entities", {})

        ok, reason = self.safety_check(intent, context)
        actions = []
        results: Dict[str, Any] = {}
        if not ok:
            results = {"status": "blocked", "reason": reason}
            self.record_interaction(user_message, intent, actions, results)
            return {"message": reason, "intent": intent, "blocked": True}

        plan = self.create_execution_plan(intent, entities)
        actions.append({"plan": plan})

        # For routing, return agent name (do not auto-execute heavy ops here)
        routed = self.route_command(intent)
        results = {"status": "planned", "route": routed, "plan": plan}

        self.record_interaction(user_message, intent, actions, results)
        # Friendly summary
        summary = self._friendly_response(intent, entities, results)
        return {"intent": intent, "confidence": understood.get("confidence"), "route": routed, "plan": plan, "summary": summary}

    def _friendly_response(self, intent: str, entities: Dict[str, Any], results: Dict[str, Any]) -> str:
        if intent == "train_model":
            return "I'll prepare a plan to improve model accuracy: review experiments, engineer features, and run hyperparameter optimization."
        if intent == "analyze_data":
            return "I'll analyze your dataset and return a risk assessment and feature recommendations."
        if intent == "deploy_model":
            return "I'll check deployment readiness and prepare a deployment package if possible."
        if intent == "autonomous_mode":
            return "Entering Autonomous Data Scientist mode: I'll run a full project pipeline and create an executive report."
        if intent == "generate_report":
            return "I'll generate a notebook and exportable Python code for your project."
        return "I've planned the requested action. Review the plan before execution."

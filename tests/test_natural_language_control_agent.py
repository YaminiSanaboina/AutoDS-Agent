import json
import os

from agents.natural_language_control_agent import NaturalLanguageControlAgent


def test_intent_detection_basic():
    agent = NaturalLanguageControlAgent(memory_path="./test_ai_memory.json")
    res = agent.understand_command("Please analyze my data and check data quality")
    assert res["intent"] == "analyze_data"
    assert res["confidence"] >= 0.8


def test_routing_map():
    agent = NaturalLanguageControlAgent(memory_path="./test_ai_memory.json")
    intent = "train_model"
    route = agent.route_command(intent)
    assert isinstance(route, str)
    assert "HyperparameterOptimizationAgent" in route or "AutonomousDataScientistAgent" not in route


def test_execution_plan_generation():
    agent = NaturalLanguageControlAgent(memory_path="./test_ai_memory.json")
    plan = agent.create_execution_plan("train_model")
    assert isinstance(plan, list)
    assert any(step["action"].lower().startswith("review") or "hyperparameter" in step["action"].lower() for step in plan)


def test_safety_checks_and_memory(tmp_path):
    memfile = tmp_path / "ai_mem.json"
    agent = NaturalLanguageControlAgent(memory_path=str(memfile))

    # Safety: trying to train without dataset
    response = agent.act_on_command("Improve accuracy", context={})
    assert response["blocked"] is True

    # Provide dataset and ensure planning proceeds
    response2 = agent.act_on_command("Improve accuracy", context={"dataset_loaded": True})
    assert response2.get("intent") in ("train_model", "optimize_hyperparameters")
    assert response2.get("plan")

    # Memory persisted
    assert os.path.exists(str(memfile))
    with open(str(memfile), "r", encoding="utf-8") as fh:
        mem = json.load(fh)
    assert isinstance(mem, list)

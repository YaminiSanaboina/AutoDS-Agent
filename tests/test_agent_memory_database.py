import json
import time
from pathlib import Path

from agents.agent_memory_database import AgentMemoryDatabase


def test_add_memory_and_retrieval(tmp_path):
    db = AgentMemoryDatabase(memory_path=str(tmp_path / "mem.json"))
    mem = db.add_memory(
        agent="DatasetIntelligenceAgent",
        category="dataset_analysis",
        title="Heart Disease Analysis",
        content="Healthcare dataset with binary classification target.",
        tags=["healthcare", "classification"],
        importance=0.92,
        success_score=0.95,
    )
    assert mem["memory_id"].startswith("MEM_")
    assert mem["agent"] == "DatasetIntelligenceAgent"
    assert len(db.memories) == 1


def test_search_memory_by_keyword_and_tags(tmp_path):
    db = AgentMemoryDatabase(memory_path=str(tmp_path / "mem2.json"))
    db.add_memory("Agent1", "dataset_analysis", "Title A", "healthcare dataset", ["health"], 0.8, 0.8)
    db.add_memory("Agent2", "model_training", "Title B", "random forest model", ["ml"], 0.6, 0.7)
    
    results = db.search_memory("healthcare", top_k=5)
    assert len(results) >= 1
    assert any(r.get("content") == "healthcare dataset" for r in results)
    
    results_tag = db.search_memory("test", tags=["health"], top_k=5)
    assert len(results_tag) >= 1


def test_get_agent_history(tmp_path):
    db = AgentMemoryDatabase(memory_path=str(tmp_path / "mem3.json"))
    db.add_memory("Agent1", "dataset_analysis", "T1", "Content1", [], 0.5, 0.5)
    db.add_memory("Agent1", "model_training", "T2", "Content2", [], 0.5, 0.5)
    db.add_memory("Agent2", "failure_analysis", "T3", "Content3", [], 0.5, 0.5)
    
    history = db.get_agent_history("Agent1")
    assert len(history) == 2
    assert all(m["agent"] == "Agent1" for m in history)


def test_delete_memory(tmp_path):
    db = AgentMemoryDatabase(memory_path=str(tmp_path / "mem4.json"))
    mem = db.add_memory("Agent1", "dataset_analysis", "Title", "Content", [], 0.5, 0.5)
    mem_id = mem["memory_id"]
    assert db.delete_memory(mem_id)
    assert len(db.memories) == 0


def test_cross_agent_learning(tmp_path):
    db = AgentMemoryDatabase(memory_path=str(tmp_path / "mem5.json"))
    mem = db.share_learning("SelfHealingAgent", "HyperparameterOptimizationAgent", "High overfitting when max_depth > 20")
    assert mem["category"] == "cross_agent_learning"
    assert "SelfHealingAgent" in mem["tags"]


def test_learn_from_experiment(tmp_path):
    db = AgentMemoryDatabase(memory_path=str(tmp_path / "mem6.json"))
    exp = {
        "agent": "ExperimentMemoryAgent",
        "title": "RandomForest",
        "result": {"best_metric": 0.92, "best_params": {"n_estimators": 100}},
        "tags": ["ml", "forest"],
        "importance": 0.7,
        "success_score": 0.9,
    }
    mem = db.learn_from_experiment(exp)
    assert mem is not None
    assert mem["category"] == "experiment_result"


def test_learn_from_failure(tmp_path):
    db = AgentMemoryDatabase(memory_path=str(tmp_path / "mem7.json"))
    fail = {
        "agent": "SelfHealingAgent",
        "title": "OOM Error",
        "root_cause": "Dataset too large",
        "solution": "Batch processing",
        "tags": ["memory", "error"],
        "importance": 0.8,
    }
    mem = db.learn_from_failure(fail)
    assert mem is not None
    assert mem["category"] == "failure_analysis"
    assert mem["success_score"] == 0.3


def test_retrieve_relevant_knowledge(tmp_path):
    db = AgentMemoryDatabase(memory_path=str(tmp_path / "mem8.json"))
    db.add_memory("Agent1", "dataset_analysis", "Title", "Use stratified split for imbalanced data", ["imbalance"], 0.8, 0.85)
    db.add_memory("Agent2", "model_training", "Title2", "Random Forest baseline", ["baseline"], 0.6, 0.7)
    
    knowledge = db.retrieve_relevant_knowledge("imbalanced healthcare dataset", top_k=5)
    assert len(knowledge) >= 1


def test_recommend_best_practices(tmp_path):
    db = AgentMemoryDatabase(memory_path=str(tmp_path / "mem9.json"))
    db.add_memory("Agent1", "best_practice", "Title", "High success practice", ["healthcare", "classification"], 0.9, 0.9)
    
    recs = db.recommend_best_practices("healthcare", "classification")
    assert len(recs) > 0


def test_generate_memory_report(tmp_path):
    db = AgentMemoryDatabase(memory_path=str(tmp_path / "mem10.json"))
    db.add_memory("Agent1", "dataset_analysis", "T1", "C1", [], 0.8, 0.85)
    db.add_memory("Agent2", "model_training", "T2", "C2", [], 0.6, 0.7)
    
    report = db.generate_memory_report()
    assert report["total_memories"] == 2
    assert report["average_success"] > 0
    assert "top_agents" in report
    assert "categories" in report
    assert "knowledge_health_score" in report


def test_memory_cleanup_on_limit(tmp_path):
    db = AgentMemoryDatabase(memory_path=str(tmp_path / "mem11.json"))
    db.MAX_MEMORIES = 5
    for i in range(10):
        db.add_memory("Agent", "dataset_analysis", f"T{i}", f"C{i}", [], 0.5 + (i * 0.05), 0.5 + (i * 0.05))
    # after cleanup, should have at most 5
    assert len(db.memories) <= 5

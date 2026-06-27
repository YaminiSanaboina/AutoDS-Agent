from agents.research_scientist_agent import ResearchScientistAgent


def test_compare_random_forest_vs_xgboost():
    agent = ResearchScientistAgent()
    result = agent.compare_algorithms("Random Forest", "XGBoost")
    assert result["winner"] in {"Random Forest", "XGBoost"}
    assert "accuracy" in result["comparison"]
    assert "speed" in result["comparison"]
    assert "interpretability" in result["comparison"]
    assert "scalability" in result["comparison"]
    assert "recommendation" in result


def test_healthcare_dataset_advice():
    agent = ResearchScientistAgent()
    dataset_report = {
        "domain": "Healthcare",
        "problem_type": "Binary Classification",
        "risks": ["Class imbalance", "Data leakage"],
        "dataset_size": 300,
        "feature_count": 15,
    }
    advice = agent.recommend_research_direction(dataset_report)
    assert "Logistic Regression" in advice["recommended_algorithms"]
    assert "Random Forest" in advice["recommended_algorithms"]
    assert "XGBoost" in advice["recommended_algorithms"]
    assert "ROC-AUC" in advice["evaluation_metrics"]


def test_explainable_ai_summary():
    agent = ResearchScientistAgent()
    summary = agent.summarize_research_topic("Explainable AI")
    assert summary["topic"] == "Explainable AI"
    assert "summary" in summary
    assert "SHAP" in summary["key_techniques"]
    assert "Healthcare" in summary["industry_applications"]


def test_experiment_plan_generation():
    agent = ResearchScientistAgent()
    plan = agent.create_experiment_plan("Improve heart disease prediction")
    assert plan["project_goal"] == "Improve heart disease prediction"
    assert isinstance(plan["experiment_plan"], list)
    assert any(phase["phase"] == "Phase 1" for phase in plan["experiment_plan"])

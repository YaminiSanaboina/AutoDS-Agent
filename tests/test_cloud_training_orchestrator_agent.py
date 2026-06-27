import os

from agents.cloud_training_orchestrator_agent import CloudTrainingOrchestratorAgent


def test_detect_system_resources():
    path = "tests/training_jobs_test.json"
    if os.path.exists(path):
        os.remove(path)

    agent = CloudTrainingOrchestratorAgent(jobs_path=path)
    resources = agent.detect_system_resources()
    assert "cpu" in resources
    assert "memory" in resources
    assert "environment" in resources
    assert resources["environment"]["python_version"]


def test_recommend_training_strategy():
    path = "tests/training_jobs_test.json"
    agent = CloudTrainingOrchestratorAgent(jobs_path=path)
    metadata = {"row_count": 50000, "column_count": 20}
    strategy = agent.recommend_training_strategy(metadata, "random_forest")
    assert strategy["recommended_infrastructure"] in {"Local CPU", "Cloud CPU", "GPU", "Distributed Cluster"}
    assert strategy["estimated_training_time_hours"] >= 0
    assert strategy["expected_cost"] >= 0


def test_estimate_training_cost():
    path = "tests/training_jobs_test.json"
    agent = CloudTrainingOrchestratorAgent(jobs_path=path)
    cost = agent.estimate_training_cost(2.5, "GPU")
    assert cost["estimated_cost"] == 5.0
    assert cost["currency"] == "USD"


def test_job_lifecycle():
    path = "tests/training_jobs_test.json"
    agent = CloudTrainingOrchestratorAgent(jobs_path=path)
    job = agent.create_training_job(
        "proj_001",
        "TestModel",
        {"dataset_name": "TestSet", "row_count": 1000},
        {"recommended_infrastructure": "Local CPU"},
    )
    assert job["status"] == "Queued"

    started = agent.start_job(job["job_id"], log_message="Job started.")
    assert started["status"] == "Running"
    assert started["start_time"] is not None

    completed = agent.complete_job(job["job_id"], log_message="Job completed successfully.")
    assert completed["status"] == "Completed"
    assert completed["end_time"] is not None

    failed = agent.fail_job(job["job_id"], "Simulated failure.")
    assert failed["status"] == "Failed"

    cancelled = agent.cancel_job(job["job_id"], reason="Test cancellation.")
    assert cancelled["status"] == "Cancelled"


def test_optimize_resource_usage():
    path = "tests/training_jobs_test.json"
    agent = CloudTrainingOrchestratorAgent(jobs_path=path)
    report = agent.optimize_resource_usage()
    assert "suggestions" in report
    assert isinstance(report["suggestions"], list)
    assert "resource_snapshot" in report


def test_get_job_status():
    path = "tests/training_jobs_test.json"
    agent = CloudTrainingOrchestratorAgent(jobs_path=path)
    jobs = agent._load_store()["jobs"]
    assert jobs
    status = agent.get_job_status(jobs[-1]["job_id"])
    assert status is not None
    assert "status" in status

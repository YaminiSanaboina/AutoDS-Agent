import json
from pathlib import Path

from agents.observability_monitoring_agent import ObservabilityMonitoringAgent


def test_record_event_creation(tmp_path):
    agent = ObservabilityMonitoringAgent(observability_path=str(tmp_path / "obs.json"))
    event = agent.record_event(
        source="TestAgent",
        event_type="MODEL_TRAINING",
        status="SUCCESS",
        severity="INFO",
        message="Training completed",
        metadata={"accuracy": 0.95}
    )
    assert event["event_id"].startswith("EVT_")
    assert event["source"] == "TestAgent"
    assert len(agent.events) == 1


def test_record_agent_metrics(tmp_path):
    agent = ObservabilityMonitoringAgent(observability_path=str(tmp_path / "obs2.json"))
    metrics = agent.record_agent_metrics(
        "Agent1",
        success_count=10,
        failure_count=2,
        execution_times=[0.5, 0.6, 0.55]
    )
    assert metrics["success_count"] == 10
    assert metrics["failure_count"] == 2
    assert abs(metrics["success_rate"] - 83.333) < 0.01
    assert metrics["avg_execution_time"] > 0


def test_record_api_call(tmp_path):
    agent = ObservabilityMonitoringAgent(observability_path=str(tmp_path / "obs3.json"))
    metrics = agent.record_api_call("/api/v1/datasets", 200, 0.123)
    assert metrics["endpoint"] == "/api/v1/datasets"
    assert metrics["call_count"] == 1
    assert metrics["success_count"] == 1
    assert metrics["success_rate"] == 100.0


def test_record_model_event(tmp_path):
    agent = ObservabilityMonitoringAgent(observability_path=str(tmp_path / "obs4.json"))
    metrics = agent.record_model_event("MODEL_001", "TRAINING", "Random Forest")
    assert metrics["training_count"] == 1
    assert "TRAINING" in metrics["last_event"]
    
    metrics = agent.record_model_event("MODEL_001", "DEPLOYMENT")
    assert metrics["deployments"] == 1


def test_calculate_system_health(tmp_path):
    agent = ObservabilityMonitoringAgent(observability_path=str(tmp_path / "obs5.json"))
    agent.record_agent_metrics("Agent1", success_count=100, failure_count=5)
    agent.record_api_call("/api/test", 200, 0.1)
    
    health = agent.calculate_system_health()
    assert "score" in health
    assert 0 <= health["score"] <= 100
    assert "status" in health
    assert "components" in health


def test_generate_alerts(tmp_path):
    agent = ObservabilityMonitoringAgent(observability_path=str(tmp_path / "obs6.json"))
    # simulate high failure rate
    agent.record_agent_metrics("FailingAgent", success_count=5, failure_count=20)
    
    alerts = agent.generate_alerts()
    assert any("FailingAgent" in str(a) for a in alerts)


def test_search_events(tmp_path):
    agent = ObservabilityMonitoringAgent(observability_path=str(tmp_path / "obs7.json"))
    agent.record_event("Agent1", "MODEL_TRAINING", "SUCCESS", "INFO", "msg1")
    agent.record_event("Agent1", "DEPLOYMENT", "SUCCESS", "INFO", "msg2")
    agent.record_event("Agent2", "ERROR", "FAILURE", "HIGH", "msg3")
    
    results = agent.search_events(source="Agent1")
    assert len(results) >= 2
    
    results = agent.search_events(severity="HIGH")
    assert len(results) >= 1


def test_get_recent_activity(tmp_path):
    agent = ObservabilityMonitoringAgent(observability_path=str(tmp_path / "obs8.json"))
    for i in range(5):
        agent.record_event(f"Agent{i}", "AGENT_EXECUTION", "SUCCESS")
    
    recent = agent.get_recent_activity(limit=3)
    assert len(recent) == 3


def test_detect_anomalies(tmp_path):
    agent = ObservabilityMonitoringAgent(observability_path=str(tmp_path / "obs9.json"))
    # create spike in failures
    for i in range(30):
        agent.record_event("Agent1", "AGENT_EXECUTION", "FAILURE")
    
    anomalies = agent.detect_anomalies()
    assert any("failure_spike" in str(a) for a in anomalies)


def test_generate_dashboard_metrics(tmp_path):
    agent = ObservabilityMonitoringAgent(observability_path=str(tmp_path / "obs10.json"))
    agent.record_event("Agent1", "MODEL_TRAINING", "SUCCESS")
    agent.record_event("Agent1", "MODEL_TRAINING", "SUCCESS")
    agent.record_event("Agent2", "DEPLOYMENT", "FAILURE")
    agent.record_agent_metrics("Agent1", success_count=50, failure_count=2)
    
    dashboard = agent.generate_dashboard_metrics()
    assert "total_events" in dashboard
    assert "system_health" in dashboard
    assert "successful_jobs" in dashboard
    assert "failed_jobs" in dashboard
    assert dashboard["total_events"] >= 3


def test_event_limit_enforcement(tmp_path):
    agent = ObservabilityMonitoringAgent(observability_path=str(tmp_path / "obs11.json"))
    agent.MAX_EVENTS = 100
    
    # create more events than limit
    for i in range(150):
        agent.record_event(f"Agent{i % 10}", "AGENT_EXECUTION", "SUCCESS")
    
    # should be trimmed to max
    assert len(agent.events) <= agent.MAX_EVENTS

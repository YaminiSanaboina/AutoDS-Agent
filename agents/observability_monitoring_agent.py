from __future__ import annotations

import datetime
import json
import os
import statistics
import uuid
from typing import Any, Dict, List, Optional


class ObservabilityMonitoringAgent:
    """Real-time monitoring and observability for the AutoDS system."""

    DEFAULT_OBSERVABILITY_FILE = "system_observability.json"
    MAX_EVENTS = 1000000
    MAX_METRICS = 100000

    EVENT_TYPES = {
        "AGENT_EXECUTION", "MODEL_TRAINING", "API_REQUEST", "DATASET_UPLOAD",
        "DEPLOYMENT", "SECURITY_EVENT", "PLUGIN_EXECUTION", "ERROR", "WARNING",
        "DATA_DRIFT", "RETRAINING", "ROLLBACK"
    }

    SEVERITIES = {"INFO", "WARNING", "HIGH", "CRITICAL"}

    def __init__(self, observability_path: Optional[str] = None):
        self.observability_path = observability_path or self.DEFAULT_OBSERVABILITY_FILE
        self._ensure_storage()
        self._load_data()

    def _now(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def _ensure_storage(self) -> None:
        if not os.path.exists(self.observability_path):
            initial = {
                "events": [],
                "agent_metrics": {},
                "api_metrics": {},
                "model_metrics": {},
                "alerts": []
            }
            with open(self.observability_path, "w", encoding="utf-8") as fh:
                json.dump(initial, fh, indent=2)

    def _load_data(self) -> None:
        try:
            with open(self.observability_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                self.events = data.get("events", [])
                self.agent_metrics = data.get("agent_metrics", {})
                self.api_metrics = data.get("api_metrics", {})
                self.model_metrics = data.get("model_metrics", {})
                self.alerts = data.get("alerts", [])
        except Exception:
            self.events = []
            self.agent_metrics = {}
            self.api_metrics = {}
            self.model_metrics = {}
            self.alerts = []

    def _save_data(self) -> None:
        data = {
            "events": self.events,
            "agent_metrics": self.agent_metrics,
            "api_metrics": self.api_metrics,
            "model_metrics": self.model_metrics,
            "alerts": self.alerts
        }
        with open(self.observability_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        self._cleanup_if_needed()

    def record_event(self, source: str, event_type: str, status: str, severity: str = "INFO",
                     message: str = "", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Record a system event."""
        event_id = f"EVT_{uuid.uuid4().hex[:8]}"
        if event_type not in self.EVENT_TYPES:
            event_type = "WARNING"
        if severity not in self.SEVERITIES:
            severity = "INFO"

        event = {
            "event_id": event_id,
            "timestamp": self._now(),
            "source": source,
            "event_type": event_type,
            "status": status,
            "severity": severity,
            "message": message,
            "metadata": metadata or {}
        }
        self.events.append(event)
        self._save_data()
        return event

    def record_agent_metrics(self, agent_name: str, success_count: int = 0, failure_count: int = 0,
                            execution_times: Optional[List[float]] = None) -> Dict[str, Any]:
        """Record agent performance metrics."""
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = {
                "agent": agent_name,
                "success_count": 0,
                "failure_count": 0,
                "execution_times": []
            }

        metrics = self.agent_metrics[agent_name]
        metrics["success_count"] += success_count
        metrics["failure_count"] += failure_count
        if execution_times:
            metrics["execution_times"].extend(execution_times)

        total = metrics["success_count"] + metrics["failure_count"]
        metrics["success_rate"] = (metrics["success_count"] / total * 100) if total > 0 else 0
        metrics["avg_execution_time"] = statistics.mean(metrics["execution_times"]) if metrics["execution_times"] else 0

        self._save_data()
        return metrics

    def record_api_call(self, endpoint: str, status_code: int, response_time: float = 0.0) -> Dict[str, Any]:
        """Record API call metrics."""
        if endpoint not in self.api_metrics:
            self.api_metrics[endpoint] = {
                "endpoint": endpoint,
                "call_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "response_times": []
            }

        metrics = self.api_metrics[endpoint]
        metrics["call_count"] += 1
        if 200 <= status_code < 300:
            metrics["success_count"] += 1
        else:
            metrics["failure_count"] += 1
        metrics["response_times"].append(response_time)

        metrics["success_rate"] = (metrics["success_count"] / metrics["call_count"] * 100) if metrics["call_count"] > 0 else 0
        metrics["avg_response_time"] = statistics.mean(metrics["response_times"]) if metrics["response_times"] else 0

        self._save_data()
        return metrics

    def record_model_event(self, model_id: str, event: str, details: str = "") -> Dict[str, Any]:
        """Record model lifecycle events."""
        if model_id not in self.model_metrics:
            self.model_metrics[model_id] = {
                "model_id": model_id,
                "training_count": 0,
                "deployments": 0,
                "rollbacks": 0,
                "drift_alerts": 0,
                "last_event": None
            }

        metrics = self.model_metrics[model_id]
        if event == "TRAINING":
            metrics["training_count"] += 1
        elif event == "DEPLOYMENT":
            metrics["deployments"] += 1
        elif event == "ROLLBACK":
            metrics["rollbacks"] += 1
        elif event == "DRIFT":
            metrics["drift_alerts"] += 1

        metrics["last_event"] = f"{event}: {details}" if details else event

        self._save_data()
        return metrics

    def calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health (0-100)."""
        # Agent reliability (30%)
        agent_health = 85
        if self.agent_metrics:
            avg_success = statistics.mean(m.get("success_rate", 100) for m in self.agent_metrics.values())
            agent_health = avg_success

        # API health (20%)
        api_health = 85
        if self.api_metrics:
            avg_api_success = statistics.mean(m.get("success_rate", 100) for m in self.api_metrics.values())
            api_health = avg_api_success

        # Model health (25%)
        model_health = 85
        if self.model_metrics:
            # penalize drift and rollbacks
            health_scores = []
            for m in self.model_metrics.values():
                base = 100
                base -= m.get("drift_alerts", 0) * 2
                base -= m.get("rollbacks", 0) * 5
                health_scores.append(max(0, base))
            model_health = statistics.mean(health_scores) if health_scores else 85

        # Security health (15%)
        security_health = 90
        high_severity_events = sum(1 for e in self.events if e.get("severity") == "CRITICAL")
        security_health = max(50, 100 - (high_severity_events * 2))

        # Resource health (10%)
        resource_health = 90

        # Weighted score
        score = (
            agent_health * 0.30 +
            api_health * 0.20 +
            model_health * 0.25 +
            security_health * 0.15 +
            resource_health * 0.10
        )
        score = min(100, max(0, score))

        if score >= 90:
            status = "Excellent"
        elif score >= 75:
            status = "Good"
        elif score >= 50:
            status = "Fair"
        else:
            status = "Poor"

        issues = []
        if agent_health < 80:
            issues.append("Agent reliability below 80%")
        if api_health < 80:
            issues.append("API health below 80%")
        if security_health < 70:
            issues.append("Security concerns detected")

        return {
            "score": int(score),
            "status": status,
            "components": {
                "agent_health": int(agent_health),
                "api_health": int(api_health),
                "model_health": int(model_health),
                "security_health": int(security_health),
                "resource_health": int(resource_health)
            },
            "issues": issues
        }

    def generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate alerts based on system conditions."""
        alerts = []

        # Check agent failures
        for agent_name, metrics in self.agent_metrics.items():
            if metrics.get("success_rate", 100) < 80:
                alerts.append({
                    "severity": "HIGH",
                    "message": f"Agent {agent_name} failure rate > 20%"
                })

        # Check API failures
        for endpoint, metrics in self.api_metrics.items():
            if metrics.get("success_rate", 100) < 90:
                alerts.append({
                    "severity": "WARNING",
                    "message": f"API endpoint {endpoint} failures detected"
                })

        # Check recent critical events
        recent_critical = [e for e in self.events[-100:] if e.get("severity") == "CRITICAL"]
        if len(recent_critical) > 2:
            alerts.append({
                "severity": "HIGH",
                "message": f"Multiple critical events detected: {len(recent_critical)}"
            })

        # Check model drift
        for model_id, metrics in self.model_metrics.items():
            if metrics.get("drift_alerts", 0) > 5:
                alerts.append({
                    "severity": "HIGH",
                    "message": f"Model {model_id} experiencing frequent drift"
                })

        # Check excessive retraining
        for model_id, metrics in self.model_metrics.items():
            if metrics.get("training_count", 0) > 20:
                alerts.append({
                    "severity": "WARNING",
                    "message": f"Model {model_id} excessive retraining cycles"
                })

        self.alerts = alerts
        self._save_data()
        return alerts

    def search_events(self, source: Optional[str] = None, event_type: Optional[str] = None,
                     severity: Optional[str] = None, status: Optional[str] = None,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """Search events with filtering."""
        results = self.events
        if source:
            results = [e for e in results if e.get("source") == source]
        if event_type:
            results = [e for e in results if e.get("event_type") == event_type]
        if severity:
            results = [e for e in results if e.get("severity") == severity]
        if status:
            results = [e for e in results if e.get("status") == status]
        return results[-limit:]

    def get_recent_activity(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events."""
        return self.events[-limit:]

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in system behavior."""
        anomalies = []

        # Spike in failures
        recent_events = self.events[-100:] if len(self.events) > 100 else self.events
        failure_count = sum(1 for e in recent_events if e.get("status") == "FAILURE")
        if failure_count > 20:
            anomalies.append({
                "type": "failure_spike",
                "severity": "HIGH",
                "message": f"Failure spike detected: {failure_count} failures in recent 100 events"
            })

        # Repeated errors from agent
        for agent_name, metrics in self.agent_metrics.items():
            if metrics.get("failure_count", 0) > 50:
                anomalies.append({
                    "type": "agent_unreliable",
                    "severity": "HIGH",
                    "message": f"Agent {agent_name} has {metrics['failure_count']} failures"
                })

        # Plugin crash loop
        plugin_events = [e for e in self.events[-100:] if e.get("event_type") == "PLUGIN_EXECUTION"]
        plugin_failures = [e for e in plugin_events if e.get("status") == "FAILURE"]
        if len(plugin_failures) > len(plugin_events) * 0.5 and len(plugin_events) > 5:
            anomalies.append({
                "type": "plugin_crash_loop",
                "severity": "CRITICAL",
                "message": "High plugin failure rate detected"
            })

        return anomalies

    def generate_dashboard_metrics(self) -> Dict[str, Any]:
        """Generate aggregated metrics for dashboard."""
        total_events = len(self.events)
        successful_jobs = sum(1 for e in self.events if e.get("status") == "SUCCESS")
        failed_jobs = sum(1 for e in self.events if e.get("status") == "FAILURE")

        top_agents = sorted(self.agent_metrics.items(), key=lambda x: x[1].get("success_rate", 0), reverse=True)[:5]
        top_agent_names = [ag for ag, _ in top_agents]

        recent_alerts = self.generate_alerts()
        health = self.calculate_system_health()

        return {
            "total_events": total_events,
            "active_agents": len(self.agent_metrics),
            "successful_jobs": successful_jobs,
            "failed_jobs": failed_jobs,
            "system_health": health["score"],
            "system_status": health["status"],
            "top_agents": top_agent_names,
            "recent_alerts": recent_alerts[:10]
        }

    def _cleanup_if_needed(self) -> None:
        """Remove oldest records if limits exceeded."""
        if len(self.events) > self.MAX_EVENTS:
            self.events = self.events[-self.MAX_EVENTS:]
            with open(self.observability_path, "w", encoding="utf-8") as fh:
                data = {
                    "events": self.events,
                    "agent_metrics": self.agent_metrics,
                    "api_metrics": self.api_metrics,
                    "model_metrics": self.model_metrics,
                    "alerts": self.alerts
                }
                json.dump(data, fh, indent=2)

        # Trim agent/api/model metrics lists if needed
        for metrics in self.agent_metrics.values():
            if "execution_times" in metrics and len(metrics["execution_times"]) > 1000:
                metrics["execution_times"] = metrics["execution_times"][-1000:]
        for metrics in self.api_metrics.values():
            if "response_times" in metrics and len(metrics["response_times"]) > 1000:
                metrics["response_times"] = metrics["response_times"][-1000:]

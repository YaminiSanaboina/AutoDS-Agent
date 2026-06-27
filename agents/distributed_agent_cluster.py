import datetime
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.agent_communication_bus import AgentCommunicationBus


class DistributedAgentCluster:
    """Distributed cluster manager for AutoDS agent nodes."""

    CLUSTER_NODES_FILE = "cluster_nodes.json"
    CLUSTER_FAILURES_FILE = "cluster_failures.json"
    FAILED_NODE_THRESHOLD_SECONDS = 60

    CAPABILITY_MAP = {
        "hyperparameter_tuning": "HyperparameterOptimizationAgent",
        "dataset_analysis": "DatasetIntelligenceAgent",
        "feature_engineering": "FeatureEngineeringAgent",
        "model_training": "ModelAgent",
        "deployment": "DeploymentAgent",
        "monitoring": "ObservabilityMonitoringAgent",
        "security_scan": "SecurityAuthAgent",
        "documentation": "DocumentationAgent",
    }

    def __init__(self, nodes_path: str = CLUSTER_NODES_FILE, failures_path: str = CLUSTER_FAILURES_FILE, bus: Optional[AgentCommunicationBus] = None):
        self.nodes_path = nodes_path
        self.failures_path = failures_path
        self.bus = bus or AgentCommunicationBus()
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.failures: List[Dict[str, Any]] = []
        self.pending_tasks: Dict[str, Dict[str, Any]] = {}
        self._ensure_storage()

    def _now(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def _ensure_storage(self) -> None:
        if not os.path.exists(self.nodes_path):
            self._save_nodes()
        else:
            self._load_nodes()

        if not os.path.exists(self.failures_path):
            self._save_failures()
        else:
            self._load_failures()

    def _load_nodes(self) -> None:
        try:
            with open(self.nodes_path, "r", encoding="utf-8") as handle:
                self.nodes = json.load(handle).get("nodes", {})
        except (FileNotFoundError, json.JSONDecodeError):
            self.nodes = {}

    def _save_nodes(self) -> None:
        with open(self.nodes_path, "w", encoding="utf-8") as handle:
            json.dump({"nodes": self.nodes}, handle, indent=2)

    def _load_failures(self) -> None:
        try:
            with open(self.failures_path, "r", encoding="utf-8") as handle:
                self.failures = json.load(handle).get("failures", [])
        except (FileNotFoundError, json.JSONDecodeError):
            self.failures = []

    def _save_failures(self) -> None:
        with open(self.failures_path, "w", encoding="utf-8") as handle:
            json.dump({"failures": self.failures}, handle, indent=2)

    def _generate_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def register_node(
        self,
        node_id: str,
        agent_type: str,
        host: str,
        port: int,
        status: str = "online",
        cpu_usage: int = 0,
        memory_usage: int = 0,
        gpu_usage: int = 0,
        active_tasks: int = 0,
    ) -> Dict[str, Any]:
        node = {
            "node_id": node_id,
            "agent_type": agent_type,
            "host": host,
            "port": port,
            "status": status,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "gpu_usage": gpu_usage,
            "active_tasks": active_tasks,
            "last_heartbeat": self._now(),
        }
        self.nodes[node_id] = node
        self._save_nodes()
        self.broadcast_cluster_message(
            event_type="cluster.node_joined",
            message=f"Node {node_id} registered",
            payload={"node_id": node_id, "agent_type": agent_type},
        )
        return node

    def update_heartbeat(
        self,
        node_id: str,
        cpu_usage: int,
        memory_usage: int,
        gpu_usage: int,
        active_tasks: int,
        latency_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not registered")

        node = self.nodes[node_id]
        node.update(
            {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "gpu_usage": gpu_usage,
                "active_tasks": active_tasks,
                "last_heartbeat": self._now(),
                "status": "online",
                "latency_ms": latency_ms if latency_ms is not None else node.get("latency_ms", 0),
            }
        )
        self._save_nodes()
        self.broadcast_cluster_message(
            event_type="cluster.node_heartbeat",
            message=f"Heartbeat updated for {node_id}",
            payload={"node_id": node_id, "cpu_usage": cpu_usage, "memory_usage": memory_usage, "active_tasks": active_tasks},
        )
        return node

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False
        del self.nodes[node_id]
        self._save_nodes()
        self.broadcast_cluster_message(
            event_type="cluster.node_removed",
            message=f"Node {node_id} removed",
            payload={"node_id": node_id},
        )
        return True

    def _resource_score(self, node: Dict[str, Any]) -> float:
        cpu = max(0, min(100, node.get("cpu_usage", 100)))
        memory = max(0, min(100, node.get("memory_usage", 100)))
        tasks = max(0, node.get("active_tasks", 0))
        cpu_score = (100 - cpu) * 0.5
        memory_score = (100 - memory) * 0.3
        task_score = max(0, 10 - tasks) * 0.2 * 10
        return cpu_score + memory_score + task_score

    def _infer_agent_type(self, task_type: str) -> Optional[str]:
        normalized = task_type.lower().replace(" ", "_")
        return self.CAPABILITY_MAP.get(normalized)

    def schedule_task(self, task: Dict[str, Any]) -> Dict[str, str]:
        if not self.nodes:
            raise RuntimeError("No available cluster nodes")

        task_type = task.get("task_type", "unknown")
        required_agent = task.get("required_agent") or self._infer_agent_type(task_type)
        online_nodes = [node for node in self.nodes.values() if node.get("status") == "online"]
        if not online_nodes:
            raise RuntimeError("No online cluster nodes available")

        candidates: List[Tuple[float, Dict[str, Any], bool]] = []
        for node in online_nodes:
            capability_match = required_agent is None or node.get("agent_type") == required_agent
            score = self._resource_score(node)
            if capability_match:
                score += 10
            candidates.append((score, node, capability_match))

        candidates.sort(key=lambda item: (-item[2], -item[0]))
        selected = candidates[0][1]
        selected["active_tasks"] = selected.get("active_tasks", 0) + 1
        self.nodes[selected["node_id"]] = selected
        self._save_nodes()

        task_id = task.get("task_id") or self._generate_id("task")
        self.pending_tasks[task_id] = {
            "task_id": task_id,
            "task_type": task_type,
            "required_agent": required_agent,
            "assigned_node": selected["node_id"],
            "status": "assigned",
            "details": task.get("details", {}),
        }
        self.broadcast_cluster_message(
            event_type="cluster.task_scheduled",
            message=f"Task {task_id} scheduled to {selected['node_id']}",
            payload={"task_id": task_id, "assigned_node": selected["node_id"], "task_type": task_type},
        )
        return {
            "assigned_node": selected["node_id"],
            "reason": "Lowest workload with matching capability" if candidates[0][2] else "Lowest workload available node",
        }

    def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            action = task.get("action")
            if callable(action):
                result = action()
            else:
                result = {"task_type": task.get("task_type"), "status": "completed"}
            return {"task_id": task.get("task_id"), "result": result, "status": "success"}
        except Exception as exc:
            return {"task_id": task.get("task_id"), "error": str(exc), "status": "failed"}

    def execute_parallel(self, tasks: List[Dict[str, Any]], max_workers: int = 4) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        completed = 0
        failed = 0

        if not tasks:
            return {"completed": 0, "failed": 0, "results": []}

        with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as executor:
            future_to_task = {executor.submit(self._execute_task, task): task for task in tasks}
            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)
                if result.get("status") == "success":
                    completed += 1
                else:
                    failed += 1

        self.broadcast_cluster_message(
            event_type="cluster.parallel_execution",
            message=f"Executed {len(tasks)} parallel tasks",
            payload={"completed": completed, "failed": failed},
        )
        return {"completed": completed, "failed": failed, "results": results}

    def balance_cluster_load(self) -> Dict[str, Any]:
        overloaded = [node for node in self.nodes.values() if node.get("status") == "online" and (node.get("cpu_usage", 0) > 85 or node.get("active_tasks", 0) > 10)]
        idle = [node for node in self.nodes.values() if node.get("status") == "online" and node.get("cpu_usage", 0) < 50 and node.get("active_tasks", 0) < 5]
        migration_plan: List[Dict[str, Any]] = []

        idle_sorted = sorted(idle, key=lambda node: self._resource_score(node), reverse=True)
        for overloaded_node in overloaded:
            while overloaded_node.get("active_tasks", 0) > 10 and idle_sorted:
                target_node = idle_sorted[0]
                overloaded_node["active_tasks"] -= 1
                target_node["active_tasks"] += 1
                migration_plan.append({
                    "from": overloaded_node["node_id"],
                    "to": target_node["node_id"],
                    "tasks_migrated": 1,
                    "reason": "Reduce overloaded node to idle cluster member",
                })
                if target_node["active_tasks"] >= 5:
                    idle_sorted.pop(0)

        self._save_nodes()
        return {"migration_plan": migration_plan, "overloaded_nodes": [node["node_id"] for node in overloaded], "idle_nodes": [node["node_id"] for node in idle]}

    def calculate_cluster_health(
        self,
        task_success_rate: float = 0.95,
        average_latency_ms: float = 50.0,
    ) -> Dict[str, Any]:
        total_nodes = len(self.nodes)
        if total_nodes == 0:
            return {"score": 0, "status": "Offline", "issues": ["No cluster nodes registered"], "details": {}}

        online_nodes = [node for node in self.nodes.values() if node.get("status") == "online"]
        availability = len(online_nodes) / total_nodes * 100
        avg_cpu = sum(node.get("cpu_usage", 0) for node in online_nodes) / max(1, len(online_nodes))
        resource_utilization_score = max(0, 100 - avg_cpu)
        latency_score = max(0, 100 - (average_latency_ms / 2))

        score = (
            availability * 0.4
            + resource_utilization_score * 0.3
            + task_success_rate * 100 * 0.2
            + latency_score * 0.1
        )
        rounded_score = round(min(100, score), 2)

        if rounded_score >= 90:
            status = "Healthy"
        elif rounded_score >= 70:
            status = "Degraded"
        elif rounded_score >= 50:
            status = "At Risk"
        else:
            status = "Unhealthy"

        issues: List[str] = []
        if availability < 80:
            issues.append("Cluster availability below 80%")
        if avg_cpu > 90:
            issues.append("Average CPU utilization is very high")
        if task_success_rate < 0.8:
            issues.append("Task success rate is below 80%")
        if average_latency_ms > 250:
            issues.append("Communication latency is elevated")

        return {
            "score": rounded_score,
            "status": status,
            "issues": issues,
            "details": {
                "availability_percent": round(availability, 2),
                "average_cpu": round(avg_cpu, 2),
                "task_success_rate": round(task_success_rate * 100, 2),
                "average_latency_ms": average_latency_ms,
            },
        }

    def detect_failed_nodes(self) -> Dict[str, Any]:
        now = datetime.datetime.now(datetime.timezone.utc)
        failed_nodes: List[str] = []
        recovery_events: List[Dict[str, Any]] = []

        for node_id, node in list(self.nodes.items()):
            last_heartbeat = node.get("last_heartbeat")
            if not last_heartbeat:
                continue
            try:
                heartbeat_time = datetime.datetime.fromisoformat(last_heartbeat)
            except ValueError:
                heartbeat_time = now

            delta = now - heartbeat_time
            if delta.total_seconds() > self.FAILED_NODE_THRESHOLD_SECONDS and node.get("status") == "online":
                node["status"] = "offline"
                failed_nodes.append(node_id)
                event = {
                    "failure_id": self._generate_id("fail"),
                    "node_id": node_id,
                    "timestamp": self._now(),
                    "reason": "Missed heartbeat",
                    "details": {
                        "last_heartbeat": last_heartbeat,
                        "delay_seconds": int(delta.total_seconds()),
                    },
                }
                self.failures.append(event)
                recovery_events.append(event)
                self.broadcast_cluster_message(
                    event_type="cluster.node_failed",
                    message=f"Node {node_id} failed due to heartbeat timeout",
                    payload={"node_id": node_id, "delay_seconds": int(delta.total_seconds())},
                )
                self._reassign_pending_tasks(node_id)

        self._save_nodes()
        self._save_failures()
        return {"failed_nodes": failed_nodes, "recoveries": recovery_events}

    def _reassign_pending_tasks(self, failed_node_id: str) -> None:
        for task_id, task_info in list(self.pending_tasks.items()):
            if task_info.get("assigned_node") != failed_node_id:
                continue
            if task_info.get("status") != "assigned":
                continue
            task_info["status"] = "reassigning"
            reassigned = self.schedule_task({
                "task_id": task_id,
                "task_type": task_info.get("task_type", "unknown"),
                "required_agent": task_info.get("required_agent"),
                "details": task_info.get("details", {}),
            })
            task_info["assigned_node"] = reassigned["assigned_node"]
            task_info["status"] = "assigned"
            task_info["reassigned_at"] = self._now()
            self.pending_tasks[task_id] = task_info

    def broadcast_cluster_message(
        self,
        event_type: str,
        message: str,
        payload: Optional[Dict[str, Any]] = None,
        priority: str = "medium",
        target_agents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self.bus.publish_event(
            event_type=event_type,
            source_agent="DistributedAgentCluster",
            payload={"message": message, **(payload or {})},
            priority=priority,
            target_agents=target_agents,
            prevent_duplicates=False,
        )

    def recommend_scaling(self, pending_tasks: Optional[int] = None) -> Dict[str, Any]:
        online_nodes = [node for node in self.nodes.values() if node.get("status") == "online"]
        avg_cpu = sum(node.get("cpu_usage", 0) for node in online_nodes) / max(1, len(online_nodes))
        pending = pending_tasks if pending_tasks is not None else len(self.pending_tasks)

        if avg_cpu > 75 or pending > 100:
            recommendation = "scale_up"
            details = "Increase cluster capacity to support high CPU utilization or pending backlog"
        elif avg_cpu < 40 and pending < 20 and len(online_nodes) > 1:
            recommendation = "scale_down"
            details = "Reduce cluster capacity due to low usage and light task volume"
        else:
            recommendation = "maintain"
            details = "Cluster capacity is currently balanced"

        return {
            "average_cpu": round(avg_cpu, 2),
            "pending_tasks": pending,
            "recommendation": recommendation,
            "details": details,
        }

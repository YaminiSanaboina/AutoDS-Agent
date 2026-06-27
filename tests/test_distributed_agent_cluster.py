import datetime
import os
import sys
import tempfile
import time

sys.path.insert(0, os.getcwd())

from agents.distributed_agent_cluster import DistributedAgentCluster
from agents.agent_communication_bus import AgentCommunicationBus


def test_register_and_update_node():
    with tempfile.TemporaryDirectory() as tempdir:
        nodes_file = os.path.join(tempdir, "cluster_nodes.json")
        failures_file = os.path.join(tempdir, "cluster_failures.json")
        bus_file = os.path.join(tempdir, "agent_events.json")
        bus = AgentCommunicationBus(events_path=bus_file)
        cluster = DistributedAgentCluster(nodes_path=nodes_file, failures_path=failures_file, bus=bus)

        node = cluster.register_node(
            node_id="NODE001",
            agent_type="HyperparameterOptimizationAgent",
            host="localhost",
            port=9001,
            cpu_usage=10,
            memory_usage=20,
            gpu_usage=0,
            active_tasks=1,
        )

        assert node["node_id"] == "NODE001"
        assert node["status"] == "online"

        updated_node = cluster.update_heartbeat(
            node_id="NODE001",
            cpu_usage=15,
            memory_usage=25,
            gpu_usage=5,
            active_tasks=2,
            latency_ms=45,
        )

        assert updated_node["cpu_usage"] == 15
        assert updated_node["memory_usage"] == 25
        assert updated_node["gpu_usage"] == 5
        assert updated_node["active_tasks"] == 2
        assert updated_node["latency_ms"] == 45


def test_schedule_task_assigns_matching_node():
    with tempfile.TemporaryDirectory() as tempdir:
        nodes_file = os.path.join(tempdir, "cluster_nodes.json")
        failures_file = os.path.join(tempdir, "cluster_failures.json")
        bus_file = os.path.join(tempdir, "agent_events.json")
        bus = AgentCommunicationBus(events_path=bus_file)
        cluster = DistributedAgentCluster(nodes_path=nodes_file, failures_path=failures_file, bus=bus)

        cluster.register_node(
            node_id="NODE001",
            agent_type="HyperparameterOptimizationAgent",
            host="localhost",
            port=9001,
            cpu_usage=70,
            memory_usage=65,
            gpu_usage=5,
            active_tasks=4,
        )
        cluster.register_node(
            node_id="NODE002",
            agent_type="FeatureEngineeringAgent",
            host="localhost",
            port=9002,
            cpu_usage=20,
            memory_usage=30,
            gpu_usage=0,
            active_tasks=1,
        )

        result = cluster.schedule_task({"task_type": "hyperparameter_tuning", "details": {}})

        assert result["assigned_node"] == "NODE001"
        assert "matching capability" in result["reason"]


def test_execute_parallel_tasks():
    cluster = DistributedAgentCluster(nodes_path=tempfile.mktemp(), failures_path=tempfile.mktemp())

    def task_action(value):
        return lambda: {"computed": value * 2}

    tasks = [
        {"task_id": f"task_{i}", "action": task_action(i), "task_type": "dataset_analysis"}
        for i in range(5)
    ]

    result = cluster.execute_parallel(tasks, max_workers=3)

    assert result["completed"] == 5
    assert result["failed"] == 0
    assert len(result["results"]) == 5
    assert all(item["status"] == "success" for item in result["results"])


def test_balance_cluster_load_moves_tasks():
    with tempfile.TemporaryDirectory() as tempdir:
        nodes_file = os.path.join(tempdir, "cluster_nodes.json")
        failures_file = os.path.join(tempdir, "cluster_failures.json")
        cluster = DistributedAgentCluster(nodes_path=nodes_file, failures_path=failures_file)

        cluster.register_node(
            node_id="NODE001",
            agent_type="ModelAgent",
            host="localhost",
            port=9003,
            cpu_usage=95,
            memory_usage=90,
            gpu_usage=30,
            active_tasks=12,
        )
        cluster.register_node(
            node_id="NODE002",
            agent_type="ModelAgent",
            host="localhost",
            port=9004,
            cpu_usage=20,
            memory_usage=25,
            gpu_usage=0,
            active_tasks=2,
        )

        plan = cluster.balance_cluster_load()

        assert "NODE001" in plan["overloaded_nodes"]
        assert "NODE002" in plan["idle_nodes"]
        assert plan["migration_plan"]
        assert plan["migration_plan"][0]["from"] == "NODE001"
        assert plan["migration_plan"][0]["to"] == "NODE002"


def test_detect_failed_nodes_marks_offline_and_reassigns_tasks():
    with tempfile.TemporaryDirectory() as tempdir:
        nodes_file = os.path.join(tempdir, "cluster_nodes.json")
        failures_file = os.path.join(tempdir, "cluster_failures.json")
        cluster = DistributedAgentCluster(nodes_path=nodes_file, failures_path=failures_file)

        cluster.register_node(
            node_id="NODE001",
            agent_type="ModelAgent",
            host="localhost",
            port=9003,
            cpu_usage=50,
            memory_usage=50,
            gpu_usage=10,
            active_tasks=1,
        )
        cluster.register_node(
            node_id="NODE002",
            agent_type="ModelAgent",
            host="localhost",
            port=9004,
            cpu_usage=10,
            memory_usage=10,
            gpu_usage=0,
            active_tasks=0,
        )

        cluster.pending_tasks["task_1"] = {
            "task_id": "task_1",
            "task_type": "model_training",
            "required_agent": "ModelAgent",
            "assigned_node": "NODE001",
            "status": "assigned",
            "details": {},
        }

        old_hb = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=120)).isoformat()
        cluster.nodes["NODE001"]["last_heartbeat"] = old_hb
        cluster._save_nodes()

        result = cluster.detect_failed_nodes()

        assert "NODE001" in result["failed_nodes"]
        assert cluster.nodes["NODE001"]["status"] == "offline"
        assert cluster.pending_tasks["task_1"]["assigned_node"] == "NODE002"


def test_calculate_cluster_health_assesses_status():
    with tempfile.TemporaryDirectory() as tempdir:
        nodes_file = os.path.join(tempdir, "cluster_nodes.json")
        failures_file = os.path.join(tempdir, "cluster_failures.json")
        cluster = DistributedAgentCluster(nodes_path=nodes_file, failures_path=failures_file)

        cluster.register_node(
            node_id="NODE001",
            agent_type="ModelAgent",
            host="localhost",
            port=9003,
            cpu_usage=10,
            memory_usage=15,
            gpu_usage=0,
            active_tasks=1,
        )
        cluster.register_node(
            node_id="NODE002",
            agent_type="ModelAgent",
            host="localhost",
            port=9004,
            cpu_usage=20,
            memory_usage=20,
            gpu_usage=0,
            active_tasks=0,
        )

        health = cluster.calculate_cluster_health(task_success_rate=0.98, average_latency_ms=20)

        assert health["status"] == "Healthy"
        assert health["score"] > 90


def test_recommend_scaling_triggers_scale_up_and_scale_down():
    with tempfile.TemporaryDirectory() as tempdir:
        nodes_file = os.path.join(tempdir, "cluster_nodes.json")
        failures_file = os.path.join(tempdir, "cluster_failures.json")
        cluster = DistributedAgentCluster(nodes_path=nodes_file, failures_path=failures_file)

        cluster.register_node(
            node_id="NODE001",
            agent_type="ModelAgent",
            host="localhost",
            port=9003,
            cpu_usage=80,
            memory_usage=70,
            gpu_usage=0,
            active_tasks=5,
        )

        scale_up = cluster.recommend_scaling(pending_tasks=120)
        assert scale_up["recommendation"] == "scale_up"

        cluster.update_heartbeat("NODE001", cpu_usage=20, memory_usage=20, gpu_usage=0, active_tasks=1)
        scale_down = cluster.recommend_scaling(pending_tasks=5)
        assert scale_down["recommendation"] in {"scale_down", "maintain"}

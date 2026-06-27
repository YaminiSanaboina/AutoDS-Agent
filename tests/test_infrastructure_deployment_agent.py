import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.getcwd())

from agents.infrastructure_deployment_agent import InfrastructureDeploymentAgent


def test_generate_docker_package_creates_expected_files():
    with tempfile.TemporaryDirectory() as tempdir:
        cwd = os.getcwd()
        os.chdir(tempdir)
        try:
            agent = InfrastructureDeploymentAgent(deployment_path=os.path.join(tempdir, "deployment_config.json"))
            metadata = agent.generate_docker_package(
                deployment_name="AutoDS Docker Test",
                python_version="3.12",
                port=8000,
                dependencies=["fastapi", "uvicorn", "pydantic"],
            )

            assert metadata["type"] == "docker"
            assert os.path.isdir(metadata["package_path"])

            expected_files = [
                "Dockerfile",
                "docker-compose.yml",
                "requirements.txt",
                ".dockerignore",
                "start.sh",
                "README.md",
            ]
            for filename in expected_files:
                assert os.path.exists(os.path.join(metadata["package_path"], filename))

            with open(os.path.join(metadata["package_path"], "Dockerfile"), "r", encoding="utf-8") as file:
                dockerfile = file.read()
            assert "FROM python:3.12-slim" in dockerfile
            assert "HEALTHCHECK" in dockerfile
            assert "CMD [\"uvicorn\", \"api.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]" in dockerfile
        finally:
            os.chdir(cwd)


def test_generate_kubernetes_manifests_creates_expected_files():
    with tempfile.TemporaryDirectory() as tempdir:
        cwd = os.getcwd()
        os.chdir(tempdir)
        try:
            agent = InfrastructureDeploymentAgent(deployment_path=os.path.join(tempdir, "deployment_config.json"))
            metadata = agent.generate_kubernetes_manifests(
                deployment_name="autods-service",
                replicas=2,
                cpu_limit="250m",
                memory_limit="256Mi",
                service_type="ClusterIP",
                namespace="test-ns",
            )

            assert metadata["type"] == "kubernetes"
            assert metadata["replicas"] == 2
            assert metadata["service_type"] == "ClusterIP"
            assert metadata["namespace"] == "test-ns"
            assert os.path.isdir(metadata["package_path"])

            manifest_files = [
                "namespace.yaml",
                "deployment.yaml",
                "service.yaml",
                "configmap.yaml",
                "secret.yaml",
                "hpa.yaml",
                "ingress.yaml",
            ]
            for manifest in manifest_files:
                assert os.path.exists(os.path.join(metadata["package_path"], manifest))

            with open(os.path.join(metadata["package_path"], "service.yaml"), "r", encoding="utf-8") as file:
                service_yaml = file.read()
            assert "type: ClusterIP" in service_yaml

            with open(os.path.join(metadata["package_path"], "deployment.yaml"), "r", encoding="utf-8") as file:
                deployment_yaml = file.read()
            assert "replicas: 2" in deployment_yaml
            assert "namespace: test-ns" in deployment_yaml

            with open(os.path.join(metadata["package_path"], "hpa.yaml"), "r", encoding="utf-8") as file:
                hpa_yaml = file.read()
            assert "apiVersion: autoscaling/v2" in hpa_yaml
        finally:
            os.chdir(cwd)


def test_generate_environment_config_writes_dotenv_example():
    with tempfile.TemporaryDirectory() as tempdir:
        cwd = os.getcwd()
        os.chdir(tempdir)
        try:
            agent = InfrastructureDeploymentAgent(deployment_path=os.path.join(tempdir, "deployment_config.json"))
            env_config = agent.generate_environment_config()

            assert env_config["DATABASE_URL"].startswith("postgresql://")
            assert env_config["API_SECRET"] == "your-secret-key-here"
            assert env_config["MODEL_STORAGE_PATH"] == "/app/models"
            assert env_config["OPENAI_API_KEY"] == "your-openai-key-here"
            assert os.path.exists(os.path.join(tempdir, ".env.example"))

            with open(os.path.join(tempdir, ".env.example"), "r", encoding="utf-8") as file:
                dotenv = file.read()
            assert "DATABASE_URL=" in dotenv
            assert "OPENAI_API_KEY=" in dotenv
        finally:
            os.chdir(cwd)


def test_scan_deployment_security_detects_issues_for_docker_and_kubernetes():
    with tempfile.TemporaryDirectory() as tempdir:
        cwd = os.getcwd()
        os.chdir(tempdir)
        try:
            dockerfile_path = os.path.join(tempdir, "Dockerfile")
            with open(dockerfile_path, "w", encoding="utf-8") as file:
                file.write("FROM python:latest\nRUN echo secret=supersecret\n")

            k8s_dir = os.path.join(tempdir, "k8s")
            os.makedirs(k8s_dir, exist_ok=True)
            service_path = os.path.join(k8s_dir, "service.yaml")
            with open(service_path, "w", encoding="utf-8") as file:
                file.write(
                    "apiVersion: v1\n"
                    "kind: Service\n"
                    "metadata:\n"
                    "  name: autods-service\n"
                    "spec:\n"
                    "  type: LoadBalancer\n"
                )
            deployment_path = os.path.join(k8s_dir, "deployment.yaml")
            with open(deployment_path, "w", encoding="utf-8") as file:
                file.write(
                    "apiVersion: apps/v1\n"
                    "kind: Deployment\n"
                    "metadata:\n"
                    "  name: autods\n"
                    "spec:\n"
                    "  template:\n"
                    "    spec:\n"
                    "      containers:\n"
                    "      - name: app\n"
                    "        securityContext:\n"
                    "          privileged: true\n"
                )

            agent = InfrastructureDeploymentAgent(deployment_path=os.path.join(tempdir, "deployment_config.json"))
            report = agent.scan_deployment_security(docker_path=dockerfile_path, k8s_path=k8s_dir)

            assert report["security_score"] < 100
            assert report["total_issues"] >= 1
            assert any(issue["issue"].lower().startswith("privileged") for issue in report["issues"])
            assert any("secret" in issue["issue"].lower() or "base image" in issue["issue"].lower() for issue in report["issues"])
        finally:
            os.chdir(cwd)


def test_calculate_production_readiness_returns_expected_grade():
    agent = InfrastructureDeploymentAgent(deployment_path=os.path.join(tempfile.gettempdir(), "deployment_config.json"))

    ready_report = agent.calculate_production_readiness(
        has_api=True,
        has_model_registry=True,
        has_security_config=True,
        has_monitoring=True,
        test_coverage_percent=90.0,
        has_documentation=True,
    )

    assert ready_report["overall_score"] == 100
    assert ready_report["grade"] == "Production Ready"

    development_report = agent.calculate_production_readiness(
        has_api=False,
        has_model_registry=False,
        has_security_config=False,
        has_monitoring=False,
        test_coverage_percent=40.0,
        has_documentation=False,
    )

    assert development_report["grade"] in {"Development Stage", "Not Ready"}
    assert development_report["overall_score"] < 60

import json
import os
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class InfrastructureDeploymentAgent:
    """
    Infrastructure Deployment Agent for Docker, Kubernetes, and production deployment automation.
    Generates deployment packages, manifests, security configurations, and readiness analysis.
    """

    DEPLOYMENT_PATH = "deployment_config.json"
    MAX_DEPLOYMENTS = 10000

    def __init__(self, deployment_path: str = DEPLOYMENT_PATH):
        self.deployment_path = deployment_path
        self.deployments = {}
        self.security_scans = {}
        self.readiness_reports = {}
        self._ensure_storage()

    def _now(self) -> str:
        """Return current UTC timestamp in ISO format."""
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def _ensure_storage(self):
        """Ensure deployment storage file exists."""
        try:
            if Path(self.deployment_path).exists():
                self._load_data()
            else:
                self._save_data()
        except Exception:
            self.deployments = {}
            self.security_scans = {}
            self.readiness_reports = {}

    def _load_data(self):
        """Load deployment data from JSON file."""
        try:
            with open(self.deployment_path, "r") as f:
                data = json.load(f)
                self.deployments = data.get("deployments", {})
                self.security_scans = data.get("security_scans", {})
                self.readiness_reports = data.get("readiness_reports", {})
        except Exception:
            self.deployments = {}
            self.security_scans = {}
            self.readiness_reports = {}

    def _save_data(self):
        """Save deployment data to JSON file."""
        try:
            data = {
                "deployments": self.deployments,
                "security_scans": self.security_scans,
                "readiness_reports": self.readiness_reports,
            }
            with open(self.deployment_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def generate_docker_package(
        self,
        deployment_name: str,
        python_version: str = "3.12",
        port: int = 8000,
        dependencies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate Docker deployment package with Dockerfile, docker-compose.yml, etc.
        
        Args:
            deployment_name: Name of deployment
            python_version: Python version (default 3.12)
            port: Application port (default 8000)
            dependencies: List of Python dependencies
        
        Returns:
            Docker package metadata
        """
        if dependencies is None:
            dependencies = ["fastapi", "uvicorn", "pydantic"]

        deployment_id = f"DCK_{os.urandom(4).hex()}"
        docker_dir = f"deployment/{deployment_id}"

        # Create directory structure
        os.makedirs(docker_dir, exist_ok=True)

        # Generate Dockerfile
        dockerfile_content = f"""FROM python:{python_version}-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT={port}
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{port}/health || exit 1

# Run application
EXPOSE {port}
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "{port}"]
"""

        # Generate docker-compose.yml
        docker_compose_content = f"""version: '3.8'

services:
  autods:
    build: .
    ports:
      - "{port}:{port}"
    environment:
      - LOG_LEVEL=INFO
      - PORT={port}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s
"""

        # Generate requirements.txt
        requirements_content = "\\n".join(dependencies)

        # Generate .dockerignore
        dockerignore_content = """__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.git
.gitignore
.dockerignore
docker-compose.yml
.vscode
.pytest_cache
*.log
.env
"""

        # Generate start.sh
        start_sh_content = f"""#!/bin/bash
set -e

echo "Starting AutoDS Agent..."
exec uvicorn api.main:app --host 0.0.0.0 --port {port} --reload
"""

        # Generate README.md
        readme_content = f"""# {deployment_name} - Docker Deployment

## Overview
AutoDS Agent Docker deployment package for {deployment_name}.

## Build and Run

### Build Docker Image
```bash
docker build -t autods:{deployment_id} .
```

### Run Container
```bash
docker run -p {port}:{port} autods:{deployment_id}
```

### Using Docker Compose
```bash
docker-compose up -d
```

## Health Check
```bash
curl http://localhost:{port}/health
```

## Environment Variables
- LOG_LEVEL: Logging level (INFO, DEBUG, WARNING, ERROR)
- PORT: Application port (default {port})

## Stopping the Container
```bash
docker-compose down
```
"""

        # Write files
        with open(f"{docker_dir}/Dockerfile", "w") as f:
            f.write(dockerfile_content)
        with open(f"{docker_dir}/docker-compose.yml", "w") as f:
            f.write(docker_compose_content)
        with open(f"{docker_dir}/requirements.txt", "w") as f:
            f.write(requirements_content)
        with open(f"{docker_dir}/.dockerignore", "w") as f:
            f.write(dockerignore_content)
        with open(f"{docker_dir}/start.sh", "w") as f:
            f.write(start_sh_content)
        with open(f"{docker_dir}/README.md", "w") as f:
            f.write(readme_content)

        # Store metadata
        metadata = {
            "deployment_id": deployment_id,
            "deployment_name": deployment_name,
            "type": "docker",
            "python_version": python_version,
            "port": port,
            "dependencies": dependencies,
            "created_at": self._now(),
            "package_path": docker_dir,
            "files": ["Dockerfile", "docker-compose.yml", "requirements.txt", ".dockerignore", "start.sh", "README.md"],
        }
        self.deployments[deployment_id] = metadata
        self._save_data()

        return metadata

    def generate_kubernetes_manifests(
        self,
        deployment_name: str,
        replicas: int = 3,
        cpu_limit: str = "500m",
        memory_limit: str = "512Mi",
        service_type: str = "LoadBalancer",
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """
        Generate Kubernetes deployment manifests.
        
        Args:
            deployment_name: Name of deployment
            replicas: Number of pod replicas
            cpu_limit: CPU limit per pod
            memory_limit: Memory limit per pod
            service_type: Service type (ClusterIP, LoadBalancer, NodePort)
            namespace: Kubernetes namespace
        
        Returns:
            Kubernetes deployment metadata
        """
        deployment_id = f"K8S_{os.urandom(4).hex()}"
        k8s_dir = f"deployment/{deployment_id}"
        os.makedirs(k8s_dir, exist_ok=True)

        # namespace.yaml
        namespace_yaml = f"""apiVersion: v1
kind: Namespace
metadata:
  name: {namespace}
"""

        # deployment.yaml
        deployment_yaml = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {deployment_name}
  namespace: {namespace}
  labels:
    app: {deployment_name}
    version: v1
spec:
  replicas: {replicas}
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: {deployment_name}
  template:
    metadata:
      labels:
        app: {deployment_name}
        version: v1
    spec:
      serviceAccountName: {deployment_name}
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: {deployment_name}
        image: autods:{deployment_id}:latest
        imagePullPolicy: Always
        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        env:
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: {deployment_name}-config
              key: log_level
        resources:
          limits:
            cpu: {cpu_limit}
            memory: {memory_limit}
          requests:
            cpu: 100m
            memory: 128Mi
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: false
          capabilities:
            drop:
            - ALL
"""

        # service.yaml
        service_yaml = f"""apiVersion: v1
kind: Service
metadata:
  name: {deployment_name}
  namespace: {namespace}
  labels:
    app: {deployment_name}
spec:
  type: {service_type}
  selector:
    app: {deployment_name}
  ports:
  - name: http
    port: 80
    targetPort: http
    protocol: TCP
  sessionAffinity: ClientIP
"""

        # configmap.yaml
        configmap_yaml = f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: {deployment_name}-config
  namespace: {namespace}
data:
  log_level: "INFO"
  api_workers: "4"
  model_cache_size: "512"
"""

        # secret.yaml (template - should be encrypted in practice)
        secret_yaml = f"""apiVersion: v1
kind: Secret
metadata:
  name: {deployment_name}-secrets
  namespace: {namespace}
type: Opaque
stringData:
  api_key: "CHANGE_ME"
  db_password: "CHANGE_ME"
"""

        # hpa.yaml (Horizontal Pod Autoscaler)
        hpa_yaml = f"""apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {deployment_name}-hpa
  namespace: {namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {deployment_name}
  minReplicas: {replicas}
  maxReplicas: {replicas * 3}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
"""

        # ingress.yaml
        ingress_yaml = f"""apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {deployment_name}-ingress
  namespace: {namespace}
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  rules:
  - host: {deployment_name}.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {deployment_name}
            port:
              number: 80
"""

        # Write manifests
        with open(f"{k8s_dir}/namespace.yaml", "w") as f:
            f.write(namespace_yaml)
        with open(f"{k8s_dir}/deployment.yaml", "w") as f:
            f.write(deployment_yaml)
        with open(f"{k8s_dir}/service.yaml", "w") as f:
            f.write(service_yaml)
        with open(f"{k8s_dir}/configmap.yaml", "w") as f:
            f.write(configmap_yaml)
        with open(f"{k8s_dir}/secret.yaml", "w") as f:
            f.write(secret_yaml)
        with open(f"{k8s_dir}/hpa.yaml", "w") as f:
            f.write(hpa_yaml)
        with open(f"{k8s_dir}/ingress.yaml", "w") as f:
            f.write(ingress_yaml)

        # Store metadata
        metadata = {
            "deployment_id": deployment_id,
            "deployment_name": deployment_name,
            "type": "kubernetes",
            "replicas": replicas,
            "cpu_limit": cpu_limit,
            "memory_limit": memory_limit,
            "service_type": service_type,
            "namespace": namespace,
            "created_at": self._now(),
            "package_path": k8s_dir,
            "manifests": ["namespace.yaml", "deployment.yaml", "service.yaml", "configmap.yaml", "secret.yaml", "hpa.yaml", "ingress.yaml"],
        }
        self.deployments[deployment_id] = metadata
        self._save_data()

        return metadata

    def generate_environment_config(self) -> Dict[str, str]:
        """
        Generate environment configuration template.
        
        Returns:
            Environment configuration dictionary
        """
        env_config = {
            "DATABASE_URL": "postgresql://user:password@localhost:5432/autods",
            "API_SECRET": "your-secret-key-here",
            "MODEL_STORAGE_PATH": "/app/models",
            "LOG_LEVEL": "INFO",
            "GROQ_API_KEY": "your-groq-key-here",
            "OPENAI_API_KEY": "your-openai-key-here",
            "ENVIRONMENT": "development",
            "DEBUG": "false",
            "CORS_ORIGINS": "*",
            "MAX_WORKERS": "4",
            "REQUEST_TIMEOUT": "30",
            "REDIS_URL": "redis://localhost:6379",
        }

        # Write .env.example
        with open(".env.example", "w") as f:
            for key, value in env_config.items():
                f.write(f"{key}={value}\n")

        return env_config

    def scan_deployment_security(
        self,
        docker_path: Optional[str] = None,
        k8s_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Scan deployment for security issues.
        
        Args:
            docker_path: Path to Dockerfile
            k8s_path: Path to Kubernetes manifests directory
        
        Returns:
            Security scan report with score and issues
        """
        scan_id = f"SEC_{os.urandom(4).hex()}"
        issues = []
        score = 100

        # Docker security checks
        if docker_path:
            try:
                with open(docker_path, "r") as f:
                    content = f.read().lower()

                    # Check for running as root
                    if "user root" not in content and "user 0" not in content:
                        pass
                    else:
                        issues.append(
                            {"severity": "HIGH", "issue": "Container running as root", "component": "Docker"}
                        )
                        score -= 15

                    # Check for base image version pinning
                    if "from python:latest" in content or "from python:3" in content:
                        if "from python:3.12-slim" not in content:
                            issues.append(
                                {"severity": "MEDIUM", "issue": "Base image not pinned to specific version", "component": "Docker"}
                            )
                            score -= 5

                    # Check for exposed secrets
                    if "password" in content or "secret" in content or "key" in content:
                        if "env " not in content or "arg " not in content:
                            issues.append(
                                {"severity": "HIGH", "issue": "Potential exposed secrets in Dockerfile", "component": "Docker"}
                            )
                            score -= 10
            except Exception:
                pass

        # Kubernetes security checks
        if k8s_path:
            try:
                for file in Path(k8s_path).glob("*.yaml"):
                    with open(file, "r") as f:
                        content = f.read()

                        # Check for privileged containers
                        if "privileged: true" in content:
                            issues.append(
                                {"severity": "CRITICAL", "issue": "Privileged container found in Kubernetes", "component": "Kubernetes"}
                            )
                            score -= 20

                        # Check for missing resource limits
                        if "limits:" not in content or "memory:" not in content:
                            issues.append(
                                {"severity": "MEDIUM", "issue": f"Missing resource limits in {file.name}", "component": "Kubernetes"}
                            )
                            score -= 5

                        # Check for security context
                        if "securityContext:" not in content:
                            issues.append(
                                {"severity": "MEDIUM", "issue": f"Missing security context in {file.name}", "component": "Kubernetes"}
                            )
                            score -= 5

                        # Check for public services
                        if "kind: Service" in content and "type: LoadBalancer" in content:
                            if "authentication" not in content.lower():
                                issues.append(
                                    {"severity": "MEDIUM", "issue": "LoadBalancer service without authentication check", "component": "Kubernetes"}
                                )
                                score -= 5
            except Exception:
                pass

        score = max(0, score)

        report = {
            "scan_id": scan_id,
            "timestamp": self._now(),
            "security_score": score,
            "total_issues": len(issues),
            "issues": issues,
            "status": "PASS" if score >= 80 else "WARN" if score >= 60 else "FAIL",
        }

        self.security_scans[scan_id] = report
        self._save_data()

        return report

    def calculate_production_readiness(
        self,
        has_api: bool = False,
        has_model_registry: bool = False,
        has_security_config: bool = False,
        has_monitoring: bool = False,
        test_coverage_percent: float = 0.0,
        has_documentation: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate production readiness score.
        
        Args:
            has_api: API availability
            has_model_registry: Model registry configured
            has_security_config: Security configuration in place
            has_monitoring: Monitoring enabled
            test_coverage_percent: Test coverage percentage (0-100)
            has_documentation: Documentation available
        
        Returns:
            Readiness report with score and grade
        """
        readiness_id = f"RDY_{os.urandom(4).hex()}"
        score = 0

        # Scoring breakdown (total 100)
        checks = {
            "api_availability": (has_api, 20),
            "model_registry": (has_model_registry, 15),
            "security_config": (has_security_config, 25),
            "monitoring_enabled": (has_monitoring, 20),
            "test_coverage": (test_coverage_percent >= 80, 15),  # >=80% coverage = pass
            "documentation": (has_documentation, 5),
        }

        for check, (condition, weight) in checks.items():
            if condition:
                score += weight

        # Add test coverage partial credit
        if "test_coverage" in checks:
            if test_coverage_percent > 0 and test_coverage_percent < 80:
                score += (test_coverage_percent / 80) * checks["test_coverage"][1]

        score = min(100, score)

        # Determine grade
        if score >= 90:
            grade = "Production Ready"
            status = "APPROVED"
        elif score >= 70:
            grade = "Near Production Ready"
            status = "REVIEW"
        elif score >= 50:
            grade = "Development Stage"
            status = "PENDING"
        else:
            grade = "Not Ready"
            status = "REJECTED"

        report = {
            "readiness_id": readiness_id,
            "timestamp": self._now(),
            "overall_score": round(score, 2),
            "grade": grade,
            "status": status,
            "checks": {
                "api_availability": has_api,
                "model_registry": has_model_registry,
                "security_config": has_security_config,
                "monitoring_enabled": has_monitoring,
                "test_coverage_percent": test_coverage_percent,
                "documentation": has_documentation,
            },
            "recommendations": self._generate_readiness_recommendations(checks),
        }

        self.readiness_reports[readiness_id] = report
        self._save_data()

        return report

    def _generate_readiness_recommendations(self, checks: Dict[str, tuple]) -> List[str]:
        """Generate recommendations based on failed checks."""
        recommendations = []

        if not checks["api_availability"][0]:
            recommendations.append("Ensure FastAPI server is properly configured and tested")
        if not checks["model_registry"][0]:
            recommendations.append("Set up model registry with versioning and rollback capability")
        if not checks["security_config"][0]:
            recommendations.append("Implement security configuration (RBAC, API keys, audit logging)")
        if not checks["monitoring_enabled"][0]:
            recommendations.append("Enable observability monitoring and alerting")
        if not checks["test_coverage"][0]:
            recommendations.append("Increase test coverage to at least 80%")
        if not checks["documentation"][0]:
            recommendations.append("Create comprehensive API and deployment documentation")

        if not recommendations:
            recommendations.append("System is production-ready. Deploy with confidence!")

        return recommendations

    def get_deployment_history(self) -> List[Dict[str, Any]]:
        """Get all deployment records."""
        return list(self.deployments.values())

    def get_security_reports(self) -> List[Dict[str, Any]]:
        """Get all security scan reports."""
        return list(self.security_scans.values())

    def get_readiness_reports(self) -> List[Dict[str, Any]]:
        """Get all readiness reports."""
        return list(self.readiness_reports.values())

    def generate_infrastructure_summary(self) -> Dict[str, Any]:
        """Generate comprehensive infrastructure summary."""
        return {
            "timestamp": self._now(),
            "total_deployments": len(self.deployments),
            "docker_deployments": sum(1 for d in self.deployments.values() if d.get("type") == "docker"),
            "kubernetes_deployments": sum(1 for d in self.deployments.values() if d.get("type") == "kubernetes"),
            "total_security_scans": len(self.security_scans),
            "avg_security_score": round(
                sum(s.get("security_score", 0) for s in self.security_scans.values()) / max(1, len(self.security_scans)),
                2
            ) if self.security_scans else 0,
            "total_readiness_reports": len(self.readiness_reports),
            "production_ready_count": sum(
                1 for r in self.readiness_reports.values() if r.get("status") == "APPROVED"
            ),
            "recent_deployments": list(self.deployments.values())[-5:],
        }

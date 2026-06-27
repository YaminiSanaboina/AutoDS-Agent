import os
import io
import json
import tempfile
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

API_KEY = "testkey"

def headers():
    return {"x-api-key": API_KEY}


def test_root_and_health():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["service"] == "AutoDS API"

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "healthy"


def test_dataset_upload_and_list():
    csv = "col1,col2\n1,2\n3,4\n"
    files = {"file": ("test.csv", io.BytesIO(csv.encode()), "text/csv")}
    data = {"dataset_name": "testset", "metadata": json.dumps({"source":"unit"})}
    r = client.post("/api/v1/datasets/upload", files=files, data=data)
    assert r.status_code == 200
    payload = r.json()
    assert "dataset_id" in payload

    r = client.get("/api/v1/datasets/list")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_agent_command_and_job():
    r = client.post("/api/v1/agents/command", json={"command": "Analyze this"})
    assert r.status_code == 200
    j = r.json()
    assert "job_id" in j
    jid = j["job_id"]
    # poll job
    r = client.get(f"/api/v1/agents/jobs/{jid}")
    assert r.status_code == 200


def test_model_listing_and_rollback():
    r = client.get("/api/v1/models")
    assert r.status_code == 200


def test_project_crud_and_archive():
    r = client.post("/api/v1/projects/create", json={"name": "p1", "description": "d"})
    assert r.status_code == 200
    pid = r.json()["project_id"]
    r = client.get("/api/v1/projects/list")
    assert r.status_code == 200
    r = client.get(f"/api/v1/projects/{pid}")
    assert r.status_code == 200
    r = client.post(f"/api/v1/projects/{pid}/archive")
    assert r.status_code == 200


def test_deployment_create_and_history():
    r = client.post("/api/v1/deployment/create", json={"model_id": "M_1", "service_name": "svc"})
    assert r.status_code == 200
    did = r.json().get("deployment_id")
    r = client.get("/api/v1/deployment/history")
    assert r.status_code == 200

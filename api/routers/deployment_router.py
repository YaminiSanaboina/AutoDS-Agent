from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import json
import uuid
import time

router = APIRouter()

DEPLOYMENTS_FILE = "api/deployments.json"

def _load_deployments():
    try:
        with open(DEPLOYMENTS_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}

def _save_deployments(data):
    os.makedirs(os.path.dirname(DEPLOYMENTS_FILE) or ".", exist_ok=True)
    with open(DEPLOYMENTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

class DeploymentCreate(BaseModel):
    model_id: str
    service_name: str = "autods_service"


@router.post("/deployment/create")
async def create_deployment(req: DeploymentCreate):
    deployments = _load_deployments()
    did = f"DEP_{uuid.uuid4().hex[:8]}"
    # create a simple prediction service file
    folder = os.path.join("deployment_package", did)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "predict_service.py"), "w", encoding="utf-8") as fh:
        fh.write("from fastapi import FastAPI\napp=FastAPI()\n@app.get('/')\ndef root():\n    return {'result':'ok'}\n")
    with open(os.path.join(folder, "Dockerfile"), "w", encoding="utf-8") as fh:
        fh.write("FROM python:3.11-slim\nCOPY . /app\nWORKDIR /app\nRUN pip install fastapi uvicorn\nCMD ['uvicorn','predict_service:app','--host','0.0.0.0','--port','8000']\n")
    with open(os.path.join(folder, "requirements.txt"), "w", encoding="utf-8") as fh:
        fh.write("fastapi\nuvicorn\n")

    deployments[did] = {"deployment_id": did, "model_id": req.model_id, "service": req.service_name, "created_at": time.time(), "package_path": folder}
    _save_deployments(deployments)
    return {"deployment_id": did, "package_path": folder}


@router.get("/deployment/history")
async def deployment_history():
    return list(_load_deployments().values())

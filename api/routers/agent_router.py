from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
import uuid
import time
import json
import os
from typing import Dict, Any

router = APIRouter()

JOBS_FILE = "api/api_jobs.json"

def _load_jobs():
    try:
        with open(JOBS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data
    except Exception:
        return {}

def _save_jobs(data):
    os.makedirs(os.path.dirname(JOBS_FILE) or ".", exist_ok=True)
    with open(JOBS_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


class CommandRequest(BaseModel):
    command: str


class AutonomousRunRequest(BaseModel):
    dataset_id: str
    goal: str


def _run_nl_command(job_id: str, payload: Dict[str, Any]):
    jobs = _load_jobs()
    jobs[job_id]["status"] = "running"
    _save_jobs(jobs)
    # simulate work
    time.sleep(0.2)
    jobs[job_id]["status"] = "completed"
    jobs[job_id]["result"] = {"intent": "analyze_and_build", "agents": ["DatasetAgent", "ModelAgent"], "summary": "Completed simulated run"}
    jobs[job_id]["completed_at"] = time.time()
    _save_jobs(jobs)


def _run_autonomous(job_id: str, payload: Dict[str, Any]):
    jobs = _load_jobs()
    jobs[job_id]["status"] = "running"
    _save_jobs(jobs)
    time.sleep(0.2)
    jobs[job_id]["status"] = "completed"
    jobs[job_id]["result"] = {"analysis": "project analysis simulated", "recommendations": ["model_x"], "decision": "use_model_x"}
    jobs[job_id]["completed_at"] = time.time()
    _save_jobs(jobs)


@router.post("/agents/command")
async def agents_command(request: CommandRequest, background_tasks: BackgroundTasks):
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    jobs = _load_jobs()
    jobs[job_id] = {"id": job_id, "type": "nl_command", "status": "queued", "created_at": time.time(), "payload": request.dict()}
    _save_jobs(jobs)
    background_tasks.add_task(_run_nl_command, job_id, request.dict())
    return {"job_id": job_id, "status": "queued"}


@router.post("/agents/autonomous-run")
async def autonomous_run(request: AutonomousRunRequest, background_tasks: BackgroundTasks):
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    jobs = _load_jobs()
    jobs[job_id] = {"id": job_id, "type": "autonomous_run", "status": "queued", "created_at": time.time(), "payload": request.dict()}
    _save_jobs(jobs)
    background_tasks.add_task(_run_autonomous, job_id, request.dict())
    return {"job_id": job_id, "status": "queued"}


@router.get("/agents/jobs/{job_id}")
async def get_job(job_id: str):
    jobs = _load_jobs()
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

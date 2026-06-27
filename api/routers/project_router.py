from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import json
import uuid
import time

router = APIRouter()

PROJECTS_FILE = "api/projects.json"

def _load_projects():
    try:
        with open(PROJECTS_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}

def _save_projects(data):
    os.makedirs(os.path.dirname(PROJECTS_FILE) or ".", exist_ok=True)
    with open(PROJECTS_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

class ProjectCreate(BaseModel):
    name: str
    description: str = ""


@router.post("/projects/create")
async def create_project(req: ProjectCreate):
    projects = _load_projects()
    pid = f"PRJ_{uuid.uuid4().hex[:8]}"
    projects[pid] = {"project_id": pid, "name": req.name, "description": req.description, "created_at": time.time(), "archived": False}
    _save_projects(projects)
    return projects[pid]


@router.get("/projects/list")
async def list_projects():
    return list(_load_projects().values())


@router.get("/projects/{project_id}")
async def get_project(project_id: str):
    projects = _load_projects()
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    return projects[project_id]


@router.put("/projects/{project_id}")
async def update_project(project_id: str, payload: dict):
    projects = _load_projects()
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    projects[project_id].update(payload)
    _save_projects(projects)
    return projects[project_id]


@router.post("/projects/{project_id}/archive")
async def archive_project(project_id: str):
    projects = _load_projects()
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    projects[project_id]["archived"] = True
    _save_projects(projects)
    return {"status": "archived", "project_id": project_id}

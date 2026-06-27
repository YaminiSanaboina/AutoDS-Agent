from __future__ import annotations

import datetime
import json
import os
import uuid
from typing import Any, Dict, List, Optional


class ProjectWorkspaceAgent:
    """Agent for managing project workspaces, dataset versions, model lineage, and collaboration notes."""

    DEFAULT_WORKSPACE_FILE = "project_workspace.json"
    PROJECT_STATUSES = {"Created", "Active", "Completed", "Archived"}

    def __init__(self, workspace_path: str = DEFAULT_WORKSPACE_FILE) -> None:
        self.workspace_path = workspace_path
        self._ensure_workspace_file()

    def _ensure_workspace_file(self) -> None:
        if not os.path.exists(self.workspace_path):
            self._save_workspace({
                "projects": [],
                "datasets": [],
                "models": [],
                "timeline": [],
                "notes": [],
            })

    def _load_workspace(self) -> Dict[str, Any]:
        try:
            with open(self.workspace_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "projects": [],
                "datasets": [],
                "models": [],
                "timeline": [],
                "notes": [],
            }

    def _save_workspace(self, data: Dict[str, Any]) -> None:
        with open(self.workspace_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)

    def _now(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def _generate_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex}"

    def create_project(
        self,
        project_name: str,
        description: str,
        owner: Optional[str] = None,
    ) -> Dict[str, Any]:
        workspace = self._load_workspace()
        project_id = self._generate_id("proj")
        now = self._now()
        project = {
            "project_id": project_id,
            "project_name": project_name,
            "description": description,
            "owner": owner or "Unknown",
            "created_at": now,
            "updated_at": now,
            "status": "Created",
        }
        workspace["projects"].append(project)
        self._save_workspace(workspace)
        return project

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        workspace = self._load_workspace()
        return next((proj for proj in workspace.get("projects", []) if proj.get("project_id") == project_id), None)

    def list_projects(self) -> List[Dict[str, Any]]:
        workspace = self._load_workspace()
        return workspace.get("projects", [])

    def update_project(self, project_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
        workspace = self._load_workspace()
        project = self.get_project(project_id)
        if not project:
            return None

        allowed_fields = {"project_name", "description", "owner", "status"}
        for field, value in updates.items():
            if field == "status" and value not in self.PROJECT_STATUSES:
                continue
            if field in allowed_fields:
                project[field] = value
        project["updated_at"] = self._now()
        self._save_workspace(workspace)
        return project

    def archive_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        return self.update_project(project_id, status="Archived")

    def add_dataset_version(
        self,
        project_id: str,
        dataset_name: str,
        dataset_metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        workspace = self._load_workspace()
        if not self.get_project(project_id):
            return None

        version_id = self._generate_id("data")
        now = self._now()
        dataset = {
            "version_id": version_id,
            "project_id": project_id,
            "dataset_name": dataset_name,
            "row_count": int(dataset_metadata.get("row_count", 0)),
            "column_count": int(dataset_metadata.get("column_count", 0)),
            "schema": dataset_metadata.get("schema", {}),
            "upload_date": now,
            "changes": dataset_metadata.get("changes", "Initial version."),
        }
        workspace["datasets"].append(dataset)
        self._save_workspace(workspace)
        return dataset

    def get_dataset_history(self, project_id: str) -> List[Dict[str, Any]]:
        workspace = self._load_workspace()
        return [
            ds for ds in workspace.get("datasets", []) if ds.get("project_id") == project_id
        ]

    def track_model_lineage(
        self,
        project_id: str,
        model_id: str,
        parent_model: Optional[str] = None,
        experiment_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        workspace = self._load_workspace()
        if not self.get_project(project_id):
            return None

        now = self._now()
        existing = next(
            (entry for entry in workspace.get("models", []) if entry.get("project_id") == project_id and entry.get("model_id") == model_id),
            None,
        )

        record = {
            "project_id": project_id,
            "model_id": model_id,
            "parent_model": parent_model,
            "experiment_id": experiment_id,
            "metadata": metadata or {},
            "tracked_at": now,
        }

        if existing:
            existing["parent_model"] = parent_model or existing.get("parent_model")
            existing["experiment_id"] = experiment_id or existing.get("experiment_id")
            existing["metadata"] = {**existing.get("metadata", {}), **(metadata or {})}
            existing["tracked_at"] = now
            record = existing
        else:
            workspace["models"].append(record)

        self._save_workspace(workspace)
        return record

    def get_model_lineage(self, project_id: str) -> List[Dict[str, Any]]:
        workspace = self._load_workspace()
        return [
            model for model in workspace.get("models", []) if model.get("project_id") == project_id
        ]

    def record_project_event(
        self,
        project_id: str,
        event_type: str,
        details: str,
    ) -> Optional[Dict[str, Any]]:
        workspace = self._load_workspace()
        if not self.get_project(project_id):
            return None

        event_id = self._generate_id("event")
        event = {
            "event_id": event_id,
            "project_id": project_id,
            "event_type": event_type,
            "details": details,
            "timestamp": self._now(),
        }
        workspace["timeline"].append(event)
        workspace["timeline"] = workspace["timeline"][-10000:]
        self._save_workspace(workspace)
        return event

    def get_project_timeline(self, project_id: str) -> List[Dict[str, Any]]:
        workspace = self._load_workspace()
        return [
            event for event in workspace.get("timeline", []) if event.get("project_id") == project_id
        ]

    def calculate_project_health(self, project_id: str) -> Dict[str, Any]:
        project = self.get_project(project_id)
        if not project:
            return {"score": 0, "status": "Critical", "issues": ["Project not found."]}

        dataset_history = self.get_dataset_history(project_id)
        models = self.get_model_lineage(project_id)
        events = self.get_project_timeline(project_id)
        notes = [note for note in self._load_workspace().get("notes", []) if note.get("project_id") == project_id]

        score = 100
        issues: List[str] = []

        if not dataset_history:
            score -= 25
            issues.append("No dataset versions tracked.")
        else:
            quality_scores = [ds.get("metadata", {}).get("quality_score") for ds in dataset_history]
            if any(score_value is None for score_value in quality_scores):
                score -= 10
                issues.append("Dataset quality metadata is incomplete.")

        if not models:
            score -= 25
            issues.append("No model lineage tracked.")
        else:
            performance_scores = [entry.get("metadata", {}).get("performance", 0) for entry in models]
            if not any(performance_scores):
                score -= 10
                issues.append("Model performance data is missing.")

        if not notes or len(notes) < 2:
            score -= 15
            issues.append("Documentation and collaboration notes are limited.")

        if project.get("status") not in {"Active", "Completed"}:
            score -= 10
            issues.append("Project status is not set to Active or Completed.")

        if not any("drift" in event.get("event_type", "").lower() or "monitor" in event.get("details", "").lower() for event in events):
            score -= 10
            issues.append("Monitoring or drift detection events are not recorded.")

        score = max(0, min(100, score))
        if score >= 85:
            status = "Excellent"
        elif score >= 70:
            status = "Good"
        elif score >= 50:
            status = "Needs Attention"
        else:
            status = "Critical"

        return {"score": score, "status": status, "issues": issues}

    def add_note(
        self,
        project_id: str,
        author: str,
        content: str,
        tags: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        workspace = self._load_workspace()
        if not self.get_project(project_id):
            return None

        note_id = self._generate_id("note")
        note = {
            "note_id": note_id,
            "project_id": project_id,
            "timestamp": self._now(),
            "author": author,
            "content": content,
            "tags": tags or [],
        }
        workspace["notes"].insert(0, note)
        workspace["notes"] = workspace["notes"][:5000]
        self._save_workspace(workspace)
        return note

    def search_notes(self, query: str) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        return [
            note
            for note in self._load_workspace().get("notes", [])
            if query_lower in note.get("content", "").lower()
            or query_lower in note.get("author", "").lower()
            or any(query_lower in tag.lower() for tag in note.get("tags", []))
        ]

import os

from agents.project_workspace_agent import ProjectWorkspaceAgent


def test_create_and_get_project():
    path = "tests/project_workspace_test.json"
    if os.path.exists(path):
        os.remove(path)

    agent = ProjectWorkspaceAgent(workspace_path=path)
    project = agent.create_project("AI Initiative", "Build predictive models.", owner="Alice")
    assert project["project_name"] == "AI Initiative"
    assert project["status"] == "Created"

    loaded = agent.get_project(project["project_id"])
    assert loaded is not None
    assert loaded["owner"] == "Alice"


def test_update_and_archive_project():
    path = "tests/project_workspace_test.json"
    agent = ProjectWorkspaceAgent(workspace_path=path)
    project = agent.list_projects()[0]
    updated = agent.update_project(project["project_id"], status="Active", description="Updated description.")
    assert updated is not None
    assert updated["status"] == "Active"
    assert updated["description"] == "Updated description."

    archived = agent.archive_project(project["project_id"])
    assert archived is not None
    assert archived["status"] == "Archived"


def test_dataset_version_tracking():
    path = "tests/project_workspace_test.json"
    agent = ProjectWorkspaceAgent(workspace_path=path)
    project = agent.list_projects()[0]
    dataset = agent.add_dataset_version(
        project["project_id"],
        "Sales Data",
        {"row_count": 1000, "column_count": 12, "schema": {"order_id": "int"}, "changes": "Initial ingest."},
    )
    assert dataset is not None
    history = agent.get_dataset_history(project["project_id"])
    assert len(history) >= 1
    assert history[0]["dataset_name"] == "Sales Data"


def test_model_lineage_tracking():
    path = "tests/project_workspace_test.json"
    agent = ProjectWorkspaceAgent(workspace_path=path)
    project = agent.list_projects()[0]
    lineage = agent.track_model_lineage(
        project["project_id"],
        "model_1",
        parent_model=None,
        experiment_id="exp_123",
        metadata={"performance": 0.88, "hyperparameters": {"max_depth": 5}},
    )
    assert lineage is not None
    assert lineage["model_id"] == "model_1"

    lineage_history = agent.get_model_lineage(project["project_id"])
    assert any(item["model_id"] == "model_1" for item in lineage_history)


def test_project_timeline_and_health():
    path = "tests/project_workspace_test.json"
    agent = ProjectWorkspaceAgent(workspace_path=path)
    project = agent.list_projects()[0]
    event = agent.record_project_event(project["project_id"], "Data cleaned", "Imputed missing values.")
    assert event is not None

    timeline = agent.get_project_timeline(project["project_id"])
    assert any(item["event_type"] == "Data cleaned" for item in timeline)

    health = agent.calculate_project_health(project["project_id"])
    assert 0 <= health["score"] <= 100
    assert health["status"] in {"Excellent", "Good", "Needs Attention", "Critical"}


def test_notes_and_search():
    path = "tests/project_workspace_test.json"
    agent = ProjectWorkspaceAgent(workspace_path=path)
    project = agent.list_projects()[0]
    note = agent.add_note(project["project_id"], "Bob", "Discussed feature selection.", tags=["discussion", "features"])
    assert note is not None
    results = agent.search_notes("feature")
    assert any("Discussed feature selection." in item["content"] for item in results)

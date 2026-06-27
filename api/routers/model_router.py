from fastapi import APIRouter, HTTPException
import json
import os

router = APIRouter()

MODELS_FILE = "storage/registry/model_registry.json"


def _load_models():
    try:
        with open(MODELS_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_models(data):
    with open(MODELS_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


@router.get("/models")
async def list_models():
    return list(_load_models().values())


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    models = _load_models()
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    return models[model_id]


@router.post("/models/{model_id}/rollback")
async def rollback_model(model_id: str):
    models = _load_models()
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    model = models[model_id]
    versions = model.get("versions", [])
    if len(versions) < 2:
        raise HTTPException(status_code=400, detail="No previous version to rollback to")
    # simple rollback: pop latest
    versions.pop()
    model["current_version"] = versions[-1]["version"]
    models[model_id] = model
    _save_models(models)
    return {"status": "rolled_back", "model_id": model_id}

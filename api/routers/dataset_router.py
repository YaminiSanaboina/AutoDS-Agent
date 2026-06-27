from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from pydantic import BaseModel
import pandas as pd
import io
import os
import json
import uuid
from typing import Optional

router = APIRouter()

DATASETS_FILE = "api/datasets.json"

def _load_datasets():
    try:
        with open(DATASETS_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}

def _save_datasets(data):
    os.makedirs(os.path.dirname(DATASETS_FILE) or ".", exist_ok=True)
    with open(DATASETS_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

class UploadResponse(BaseModel):
    dataset_id: str
    rows: int
    columns: int
    status: str


@router.post("/datasets/upload", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: str = Form(...),
    metadata: Optional[str] = Form(None),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {exc}")

    dataset_id = f"DS_{uuid.uuid4().hex[:8]}"
    record = {
        "dataset_id": dataset_id,
        "name": dataset_name,
        "filename": file.filename,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "uploaded_at": pd.Timestamp.utcnow().isoformat(),
        "metadata": metadata,
        "statistics": df.describe(include='all').to_dict(),
    }

    store = _load_datasets()
    store[dataset_id] = record
    _save_datasets(store)

    return {"dataset_id": dataset_id, "rows": record["rows"], "columns": record["columns"], "status": "uploaded"}


@router.get("/datasets/get/{dataset_id}")
async def get_dataset(dataset_id: str):
    store = _load_datasets()
    if dataset_id not in store:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return store[dataset_id]


@router.get("/datasets/list")
async def list_datasets():
    store = _load_datasets()
    return list(store.values())

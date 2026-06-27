"""Sample dataset catalog for Data Hub."""

import os

import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _ensure_sklearn_csvs():
    os.makedirs(DATA_DIR, exist_ok=True)
    specs = {
        "Iris.csv": (load_iris, True),
        "Wine-Quality.csv": (load_wine, True),
        "Heart-Disease.csv": (load_breast_cancer, True),
    }
    for filename, (loader, _) in specs.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            data = loader(as_frame=True)
            df = data.frame
            df["target"] = data.target
            df.to_csv(path, index=False)


def get_sample_datasets():
    _ensure_sklearn_csvs()
    return [
        {
            "id": "titanic",
            "name": "Titanic Dataset",
            "filename": "Titanic-Dataset.csv",
            "path": os.path.join(DATA_DIR, "Titanic-Dataset.csv"),
            "description": "Passenger survival records with demographics and ticket class.",
            "use_case": "Binary classification — predict survival",
            "icon": "🚢",
        },
        {
            "id": "churn",
            "name": "Customer Churn Dataset",
            "filename": "WA_Fn-UseC_-Telco-Customer-Churn.csv",
            "path": os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
            "description": "Telecom customer profiles, services, and churn labels.",
            "use_case": "Customer retention & churn prediction",
            "icon": "📞",
        },
        {
            "id": "housing",
            "name": "Housing Price Dataset",
            "filename": "Housing.csv",
            "path": os.path.join(DATA_DIR, "Housing.csv"),
            "description": "Residential property attributes and sale prices.",
            "use_case": "Regression — price prediction",
            "icon": "🏠",
        },
        {
            "id": "iris",
            "name": "Iris Dataset",
            "filename": "Iris.csv",
            "path": os.path.join(DATA_DIR, "Iris.csv"),
            "description": "Classic flower measurements across three species.",
            "use_case": "Multi-class classification",
            "icon": "🌸",
        },
        {
            "id": "wine",
            "name": "Wine Quality Dataset",
            "filename": "Wine-Quality.csv",
            "path": os.path.join(DATA_DIR, "Wine-Quality.csv"),
            "description": "Chemical properties of wine samples by cultivar.",
            "use_case": "Classification & clustering",
            "icon": "🍷",
        },
        {
            "id": "heart",
            "name": "Heart Disease Dataset",
            "filename": "Heart-Disease.csv",
            "path": os.path.join(DATA_DIR, "Heart-Disease.csv"),
            "description": "Medical diagnostic features for malignant vs benign cases.",
            "use_case": "Medical diagnosis classification",
            "icon": "❤️",
        },
    ]


def load_sample_dataset(dataset_id):
    for ds in get_sample_datasets():
        if ds["id"] == dataset_id:
            return pd.read_csv(ds["path"]), ds["filename"]
    raise ValueError(f"Unknown dataset: {dataset_id}")

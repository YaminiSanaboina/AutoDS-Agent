import os
import tempfile
import sys
import json

sys.path.insert(0, os.getcwd())

import pandas as pd
from agents.drift_monitoring_agent import DataDriftMonitoringAgent


def _create_dataframe(data):
    return pd.DataFrame(data)


def test_no_drift_data():
    reference = _create_dataframe({
        "age": [25, 30, 35, 40],
        "income": [50, 60, 70, 80],
        "segment": ["A", "B", "A", "B"],
    })
    new_data = reference.copy()
    agent = DataDriftMonitoringAgent(reference_path=os.path.join(tempfile.gettempdir(), "reference.json"), history_path=os.path.join(tempfile.gettempdir(), "drift_history.json"))
    reference_data = agent.register_reference_data(reference, dataset_name="test_dataset")
    drift_report = agent.generate_drift_report(reference_data, new_data)
    assert drift_report["severity"] == "Low"
    assert drift_report["drift_score"] == 0.0
    assert drift_report["data_quality"]["health_score"] == 100
    print("PASS: No drift data")


def test_slight_numerical_shift():
    reference = _create_dataframe({
        "age": [25, 30, 35, 40, 45],
        "income": [50, 60, 70, 80, 90],
    })
    new_data = _create_dataframe({
        "age": [26, 31, 36, 41, 46],
        "income": [55, 65, 75, 85, 95],
    })
    agent = DataDriftMonitoringAgent(reference_path=os.path.join(tempfile.gettempdir(), "reference2.json"), history_path=os.path.join(tempfile.gettempdir(), "drift_history2.json"))
    reference_data = agent.register_reference_data(reference, dataset_name="shift_dataset")
    drift_report = agent.generate_drift_report(reference_data, new_data)
    assert drift_report["severity"] in {"Medium", "High"}
    assert drift_report["recommended_action"] != "Continue monitoring."
    print("PASS: Slight numerical shift")


def test_major_distribution_change():
    reference = _create_dataframe({
        "age": [20, 22, 24, 26, 28, 30, 32, 34],
        "income": [40, 45, 50, 55, 60, 65, 70, 75],
    })
    new_data = _create_dataframe({
        "age": [60, 62, 64, 66, 68, 70, 72, 74],
        "income": [120, 125, 130, 135, 140, 145, 150, 155],
    })
    agent = DataDriftMonitoringAgent(reference_path=os.path.join(tempfile.gettempdir(), "reference3.json"), history_path=os.path.join(tempfile.gettempdir(), "drift_history3.json"))
    reference_data = agent.register_reference_data(reference, dataset_name="major_shift_dataset")
    drift_report = agent.generate_drift_report(reference_data, new_data)
    assert drift_report["severity"] == "High"
    assert "Retrain" in drift_report["recommended_action"]
    print("PASS: Major distribution change")


def test_new_categories_warning():
    reference = _create_dataframe({
        "product": ["A", "B", "A", "B"],
        "price": [10, 15, 10, 15],
    })
    new_data = _create_dataframe({
        "product": ["A", "C", "A", "C"],
        "price": [10, 15, 10, 15],
    })
    agent = DataDriftMonitoringAgent(reference_path=os.path.join(tempfile.gettempdir(), "reference4.json"), history_path=os.path.join(tempfile.gettempdir(), "drift_history4.json"))
    reference_data = agent.register_reference_data(reference, dataset_name="category_change_dataset")
    drift_report = agent.generate_drift_report(reference_data, new_data)
    alert = agent.generate_alert(drift_report)
    assert "HIGH ALERT" == alert["level"] or "WARNING" == alert["level"]
    assert "product" in drift_report["features_affected"]
    print("PASS: New categories warning")


def test_missing_columns_data_quality_issue():
    reference = _create_dataframe({
        "age": [25, 35, 45],
        "income": [50, 70, 90],
        "gender": ["M", "F", "M"],
    })
    new_data = _create_dataframe({
        "age": [26, 36, 46],
        "income": [55, 75, 95],
    })
    agent = DataDriftMonitoringAgent(reference_path=os.path.join(tempfile.gettempdir(), "reference5.json"), history_path=os.path.join(tempfile.gettempdir(), "drift_history5.json"))
    reference_data = agent.register_reference_data(reference, dataset_name="missing_columns_dataset")
    drift_report = agent.generate_drift_report(reference_data, new_data)
    assert "Missing columns" in drift_report["data_quality"]["issues"][0]
    assert drift_report["data_quality"]["health_score"] < 100
    print("PASS: Missing columns data quality issue")


if __name__ == "__main__":
    test_no_drift_data()
    test_slight_numerical_shift()
    test_major_distribution_change()
    test_new_categories_warning()
    test_missing_columns_data_quality_issue()
    print("ALL drift monitoring tests passed")

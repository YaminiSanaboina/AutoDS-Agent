import json
import os
import pickle
import tempfile

from sklearn.linear_model import LinearRegression

from agents.api_deployment_agent import APIDeploymentAgent


def test_generate_fastapi_service_and_deployment_package():
    with tempfile.TemporaryDirectory() as tempdir:
        model = LinearRegression()
        X = [[1.0, 2.0], [3.0, 4.0]]
        y = [5.0, 6.0]
        model.fit(X, y)

        model_file = os.path.join(tempdir, "model.pkl")
        with open(model_file, "wb") as handle:
            pickle.dump(model, handle)

        agent = APIDeploymentAgent(deployment_dir=tempdir, history_path=os.path.join(tempdir, "deployment_history.json"))
        service_result = agent.generate_fastapi_service(model_file=model_file, feature_names=["feature1", "feature2"])

        assert service_result["status"] == "success"
        assert os.path.exists(service_result["app_file"])

        package_result = agent.build_deployment_package(model_source_path=model_file, feature_names=["feature1", "feature2"])
        assert package_result["status"] == "success"
        assert os.path.exists(package_result["app_file"])
        assert os.path.exists(package_result["requirements_file"])
        assert os.path.exists(package_result["dockerfile"])
        assert os.path.exists(package_result["readme"])
        assert os.path.exists(package_result["model_file"])


def test_calculate_deployment_readiness_scores():
    agent = APIDeploymentAgent(deployment_dir=tempfile.mkdtemp(), history_path=os.path.join(tempfile.mkdtemp(), "deployment_history.json"))

    ready = agent.calculate_deployment_readiness(
        model_quality=92.0,
        data_quality=88.0,
        explainability_available=True,
        tests_completed=True,
        drift_monitoring_configured=True,
    )
    assert ready["score"] >= 90
    assert ready["status"] == "Production Ready"

    minor = agent.calculate_deployment_readiness(
        model_quality=78.0,
        data_quality=75.0,
        explainability_available=True,
        tests_completed=False,
        drift_monitoring_configured=True,
    )
    assert 70 <= minor["score"] < 90
    assert minor["status"] == "Needs Minor Improvements"

    not_ready = agent.calculate_deployment_readiness(
        model_quality=50.0,
        data_quality=45.0,
        explainability_available=False,
        tests_completed=False,
        drift_monitoring_configured=False,
    )
    assert not_ready["score"] < 70
    assert not_ready["status"] == "Not Ready for Deployment"


def test_record_and_retrieve_deployment_history():
    with tempfile.TemporaryDirectory() as tempdir:
        history_path = os.path.join(tempdir, "deployment_history.json")
        agent = APIDeploymentAgent(deployment_dir=tempdir, history_path=history_path)
        record = agent.record_deployment_event(
            model_id="MODEL_001",
            version="MODEL_v1",
            environment="staging",
            api_url="http://localhost:8000",
            status="Deployed",
        )

        assert record["deployment_id"] == "DEPLOY_001"
        assert record["model_id"] == "MODEL_001"

        history = agent.get_deployment_history()
        assert len(history) == 1
        assert history[0]["api_url"] == "http://localhost:8000"


if __name__ == "__main__":
    test_generate_fastapi_service_and_deployment_package()
    test_calculate_deployment_readiness_scores()
    test_record_and_retrieve_deployment_history()
    print("ALL API deployment tests passed")

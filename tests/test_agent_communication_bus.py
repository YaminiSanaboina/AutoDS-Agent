import os

from agents.agent_communication_bus import AgentCommunicationBus


def test_publish_event_and_duplicate_prevention():
    path = "tests/agent_events_test.json"
    if os.path.exists(path):
        os.remove(path)

    bus = AgentCommunicationBus(events_path=path)
    event = bus.publish_event(
        event_type="dataset_uploaded",
        source_agent="DataAgent",
        payload={"source": "upload", "details": "New dataset added."},
    )
    assert event["event_type"] == "dataset_uploaded"
    assert event["status"] == "created"

    try:
        bus.publish_event(
            event_type="dataset_uploaded",
            source_agent="DataAgent",
            payload={"source": "upload", "details": "New dataset added."},
        )
        assert False, "Duplicate event should not be allowed"
    except ValueError:
        assert True


def test_subscribe_unsubscribe_and_get_subscribers():
    path = "tests/agent_events_test.json"
    bus = AgentCommunicationBus(events_path=path)
    subscription = bus.subscribe("DriftAgent", ["drift_detected"], callback=None)
    assert subscription["agent_name"] == "DriftAgent"

    subscribers = bus.get_subscribers("drift_detected")
    assert any(sub["agent_name"] == "DriftAgent" for sub in subscribers)

    removed = bus.unsubscribe("DriftAgent")
    assert removed is True
    subscribers = bus.get_subscribers("drift_detected")
    assert not any(sub["agent_name"] == "DriftAgent" for sub in subscribers)


def test_dispatch_event_and_callback_execution():
    path = "tests/agent_events_test.json"
    bus = AgentCommunicationBus(events_path=path)
    bus.subscribe("RetrainingAgent", ["retraining_required"], callback=None)
    event = bus.publish_event(
        event_type="retraining_required",
        source_agent="AutoMLAgent",
        payload={"source": "validation", "details": "Performance drift detected."},
    )
    result = bus.dispatch_event(event["event_id"])
    assert "RetrainingAgent" in result["successful_deliveries"]
    assert not result["failed_deliveries"]


def test_event_replay_and_history():
    path = "tests/agent_events_test.json"
    bus = AgentCommunicationBus(events_path=path)
    events = bus.get_event_history()
    assert events
    event_id = events[-1]["event_id"]
    replay_result = bus.replay_event(event_id)
    assert isinstance(replay_result, dict)
    history = bus.get_events_by_agent("AutoMLAgent")
    assert history


def test_dead_letter_queue_on_failed_delivery():
    path = "tests/agent_events_test.json"
    bus = AgentCommunicationBus(events_path=path)

    # subscribe with invalid callback to force failure
    bus.subscribe("BrokenAgent", ["model_trained"], callback="non_existent_callback")
    event = bus.publish_event(
        event_type="model_trained",
        source_agent="TrainingAgent",
        payload={"source": "train", "details": "Model finished."},
    )
    result = bus.dispatch_event(event["event_id"])
    assert result["failed_deliveries"]
    store = bus._load_store()
    assert any(evt["event_id"] == event["event_id"] for evt in store["dead_letter_queue"])


def test_notification_summary_and_retry_failed_deliveries():
    path = "tests/agent_events_test.json"
    bus = AgentCommunicationBus(events_path=path)
    summary = bus.create_notification_summary("ethics_warning", {"source": "EthicsAgent", "details": "Bias risk increased."})
    assert "Bias risk increased" in summary

    # retry dead letter if available
    dead_events = bus._load_store()["dead_letter_queue"]
    if dead_events:
        event_id = dead_events[-1]["event_id"]
        try:
            retry_result = bus.retry_failed_deliveries(event_id)
            assert isinstance(retry_result, dict)
        except ValueError:
            assert False, "Retry should not raise if dead-letter event exists"

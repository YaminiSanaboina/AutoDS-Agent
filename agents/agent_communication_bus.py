from __future__ import annotations

import datetime
import json
import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple


class AgentCommunicationBus:
    """In-memory and persisted event bus for AutoDS agent communication."""

    DEFAULT_EVENTS_FILE = "storage/logs/agent_events.json"
    PRIORITIES = {"low", "medium", "high", "critical"}
    STATUSES = {"created", "dispatched", "processed", "failed"}
    HELPER_EVENTS = {
        "dataset_uploaded": "A new dataset has been uploaded.",
        "model_trained": "A model has completed training.",
        "drift_detected": "Data drift has been detected.",
        "retraining_triggered": "Retraining workflow has been triggered.",
        "deployment_completed": "Model deployment has completed.",
        "experiment_completed": "An experiment has completed.",
        "ethics_warning": "An ethics warning has been raised.",
    }

    def __init__(self, events_path: str = DEFAULT_EVENTS_FILE) -> None:
        self.events_path = events_path
        self._ensure_events_file()

    def _ensure_events_file(self) -> None:
        if not os.path.exists(self.events_path):
            self._save_store({
                "events": [],
                "subscriptions": [],
                "dead_letter_queue": [],
            })

    def _load_store(self) -> Dict[str, Any]:
        try:
            with open(self.events_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"events": [], "subscriptions": [], "dead_letter_queue": []}

    def _save_store(self, data: Dict[str, Any]) -> None:
        with open(self.events_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)

    def _now(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def _generate_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex}"

    def _event_fingerprint(
        self,
        event_type: str,
        source_agent: str,
        payload: Dict[str, Any],
        target_agents: Optional[List[str]] = None,
    ) -> str:
        """Stable fingerprint for duplicate detection."""
        fingerprint_data = {
            "event_type": event_type,
            "source_agent": source_agent,
            "payload": payload,
            "target_agents": sorted(target_agents or []),
        }
        return json.dumps(fingerprint_data, sort_keys=True, default=str)

    def _find_duplicate_event(
        self,
        store: Dict[str, Any],
        event_type: str,
        source_agent: str,
        payload: Dict[str, Any],
        target_agents: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        fingerprint = self._event_fingerprint(event_type, source_agent, payload, target_agents)
        for existing in store.get("events", []):
            if self._event_fingerprint(
                existing.get("event_type", ""),
                existing.get("source_agent", ""),
                existing.get("payload", {}),
                existing.get("target_agents", []),
            ) == fingerprint:
                return existing
        return None

    def publish_event(
        self,
        event_type: str,
        source_agent: str,
        payload: Dict[str, Any],
        priority: str = "medium",
        target_agents: Optional[List[str]] = None,
        prevent_duplicates: bool = True,
    ) -> Dict[str, Any]:
        priority = priority.lower()
        if priority not in self.PRIORITIES:
            priority = "medium"

        event_type = event_type.strip()
        if not event_type:
            raise ValueError("event_type must be a non-empty string")

        store = self._load_store()
        if prevent_duplicates:
            duplicate = self._find_duplicate_event(store, event_type, source_agent, payload, target_agents)
            if duplicate is not None:
                raise ValueError(
                    f"Duplicate event already exists: {duplicate.get('event_id')} "
                    f"for event_type '{event_type}' from '{source_agent}'"
                )

        event_id = self._generate_id("evt")
        event = {
            "event_id": event_id,
            "timestamp": self._now(),
            "event_type": event_type,
            "source_agent": source_agent,
            "target_agents": target_agents or [],
            "payload": payload,
            "priority": priority,
            "status": "created",
            "delivery_logs": [],
        }
        store["events"].append(event)
        store["events"] = store["events"][-10000:]
        self._save_store(store)
        return event

    def subscribe(
        self,
        agent_name: str,
        event_types: List[str],
        callback: Optional[str] = None,
    ) -> Dict[str, Any]:
        store = self._load_store()
        subscription = {
            "agent_name": agent_name,
            "event_types": event_types,
            "callback": callback,
            "subscribed_at": self._now(),
        }
        existing = next((sub for sub in store.get("subscriptions", []) if sub.get("agent_name") == agent_name), None)
        if existing:
            existing["event_types"] = list(set(existing.get("event_types", []) + event_types))
            existing["callback"] = callback or existing.get("callback")
        else:
            store["subscriptions"].append(subscription)
        self._save_store(store)
        return subscription

    def unsubscribe(self, agent_name: str) -> bool:
        store = self._load_store()
        subscriptions = [sub for sub in store.get("subscriptions", []) if sub.get("agent_name") != agent_name]
        removed = len(subscriptions) < len(store.get("subscriptions", []))
        store["subscriptions"] = subscriptions
        self._save_store(store)
        return removed

    def get_subscribers(self, event_type: str) -> List[Dict[str, Any]]:
        store = self._load_store()
        return [
            sub for sub in store.get("subscriptions", []) if event_type in sub.get("event_types", [])
        ]

    def dispatch_event(self, event_id: str) -> Dict[str, List[str]]:
        store = self._load_store()
        event = next((evt for evt in store.get("events", []) if evt.get("event_id") == event_id), None)
        if not event:
            raise ValueError("Event not found")

        subscribers = self.get_subscribers(event.get("event_type", ""))
        successful_deliveries: List[str] = []
        failed_deliveries: List[str] = []

        for subscriber in subscribers:
            agent_name = subscriber.get("agent_name")
            callback_name = subscriber.get("callback")
            try:
                if callback_name:
                    if callback_name in globals() and callable(globals()[callback_name]):
                        callback = globals()[callback_name]
                        callback(event)
                    else:
                        raise ValueError(f"Callback '{callback_name}' not found or not callable")
                successful_deliveries.append(agent_name)
                event["delivery_logs"].append({
                    "agent_name": agent_name,
                    "status": "processed",
                    "timestamp": self._now(),
                })
            except Exception as exc:
                failed_deliveries.append(agent_name)
                event["delivery_logs"].append({
                    "agent_name": agent_name,
                    "status": "failed",
                    "timestamp": self._now(),
                    "error": str(exc),
                })

        event["status"] = "dispatched"
        if failed_deliveries:
            event["status"] = "failed"
            store["dead_letter_queue"].append(event)
            store["dead_letter_queue"] = store["dead_letter_queue"][-10000:]

        self._save_store(store)
        return {"successful_deliveries": successful_deliveries, "failed_deliveries": failed_deliveries}

    def get_event_history(self) -> List[Dict[str, Any]]:
        store = self._load_store()
        return store.get("events", [])

    def get_events_by_agent(self, agent_name: str) -> List[Dict[str, Any]]:
        return [
            event for event in self.get_event_history()
            if event.get("source_agent") == agent_name or agent_name in event.get("target_agents", [])
        ]

    def replay_event(self, event_id: str) -> Dict[str, Any]:
        store = self._load_store()
        event = next((evt for evt in store.get("events", []) if evt.get("event_id") == event_id), None)
        if not event:
            raise ValueError("Event not found")
        return self.dispatch_event(event_id)

    def retry_failed_deliveries(self, event_id: str) -> Dict[str, Any]:
        store = self._load_store()
        event = next((evt for evt in store.get("dead_letter_queue", []) if evt.get("event_id") == event_id), None)
        if not event:
            raise ValueError("Failed event not found")
        self.unsubscribe(event.get("source_agent", ""))
        return self.dispatch_event(event_id)

    def create_notification_summary(self, event_type: str, payload: Dict[str, Any]) -> str:
        message = self.HELPER_EVENTS.get(event_type, "An event has occurred.")
        summary = f"{message} Source: {payload.get('source', 'unknown')} | Details: {payload.get('details', '')}"
        return summary

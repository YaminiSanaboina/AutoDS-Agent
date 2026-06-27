"""
Thread-safe execution state for background pipeline execution.
Allows background thread to write progress and results safely, while Streamlit render loop reads and syncs.
"""

import threading
from datetime import datetime
from typing import Any, Dict, List, Optional


class ThreadSafeExecutionState:
    """
    Shared state object for thread-safe communication between background pipeline and Streamlit UI.
    Uses a lock to prevent race conditions.
    """
    
    def __init__(self):
        """Initialize thread-safe execution state."""
        self._lock = threading.Lock()
        self._status = "idle"  # idle, running, completed, error
        self._current_stage = None
        self._progress = 0  # 0-100
        self._elapsed_time = 0.0
        self._start_time = None
        self._stage_results = {}  # {stage_key: result_data}
        self._stage_statuses = {}  # {stage_key: "waiting"|"active"|"completed"}
        self._events = []  # List of event dicts
        self._final_result = None
        self._error_message = None
        self._execution_state_id = None
        self._last_update_time = None
    
    def set_status(self, status: str) -> None:
        """Set overall execution status."""
        with self._lock:
            self._status = status
            self._last_update_time = datetime.utcnow()
    
    def get_status(self) -> str:
        """Get overall execution status."""
        with self._lock:
            return self._status
    
    def set_current_stage(self, stage: str) -> None:
        """Set currently executing stage."""
        with self._lock:
            self._current_stage = stage
            self._last_update_time = datetime.utcnow()
    
    def get_current_stage(self) -> Optional[str]:
        """Get currently executing stage."""
        with self._lock:
            return self._current_stage
    
    def set_progress(self, percent: int) -> None:
        """Set progress percentage (0-100)."""
        with self._lock:
            self._progress = max(0, min(100, percent))
            self._last_update_time = datetime.utcnow()
    
    def get_progress(self) -> int:
        """Get progress percentage."""
        with self._lock:
            return self._progress
    
    def set_elapsed_time(self, seconds: float) -> None:
        """Set elapsed time in seconds."""
        with self._lock:
            self._elapsed_time = seconds
            self._last_update_time = datetime.utcnow()
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        with self._lock:
            return self._elapsed_time
    
    def set_start_time(self, dt: datetime) -> None:
        """Set start time."""
        with self._lock:
            self._start_time = dt
    
    def get_start_time(self) -> Optional[datetime]:
        """Get start time."""
        with self._lock:
            return self._start_time
    
    def add_stage_result(self, stage: str, result: Any) -> None:
        """Store result from a completed stage."""
        with self._lock:
            self._stage_results[stage] = result
            self._last_update_time = datetime.utcnow()
    
    def get_stage_result(self, stage: str) -> Optional[Any]:
        """Get result for a specific stage."""
        with self._lock:
            return self._stage_results.get(stage)
    
    def get_all_stage_results(self) -> Dict[str, Any]:
        """Get all stage results."""
        with self._lock:
            return dict(self._stage_results)
    
    def set_stage_status(self, stage: str, status: str) -> None:
        """Set status for a specific stage (waiting, active, completed)."""
        with self._lock:
            self._stage_statuses[stage] = status
            self._last_update_time = datetime.utcnow()
    
    def get_stage_status(self, stage: str) -> str:
        """Get status for a specific stage."""
        with self._lock:
            return self._stage_statuses.get(stage, "waiting")
    
    def get_all_stage_statuses(self) -> Dict[str, str]:
        """Get all stage statuses."""
        with self._lock:
            return dict(self._stage_statuses)
    
    def add_event(self, event: Dict[str, Any]) -> None:
        """Add event to history (with timestamp)."""
        with self._lock:
            event_with_ts = dict(event)
            event_with_ts["timestamp"] = datetime.utcnow().isoformat()
            self._events.append(event_with_ts)
            # Keep only last 50 events
            if len(self._events) > 50:
                self._events = self._events[-50:]
            self._last_update_time = datetime.utcnow()
    
    def get_events(self) -> List[Dict[str, Any]]:
        """Get event history."""
        with self._lock:
            return list(self._events)
    
    def set_final_result(self, result: Any) -> None:
        """Set final pipeline result."""
        with self._lock:
            self._final_result = result
            self._last_update_time = datetime.utcnow()
    
    def get_final_result(self) -> Optional[Any]:
        """Get final pipeline result."""
        with self._lock:
            return self._final_result

    def clear_final_result(self) -> None:
        """Clear final result after it has been applied to session state."""
        with self._lock:
            self._final_result = None
            self._last_update_time = datetime.utcnow()

    def set_execution_state_id(self, execution_state_id: str) -> None:
        """Set the background execution state identifier."""
        with self._lock:
            self._execution_state_id = execution_state_id
            self._last_update_time = datetime.utcnow()

    def get_execution_state_id(self) -> Optional[str]:
        """Get the background execution state identifier."""
        with self._lock:
            return self._execution_state_id

    def set_error(self, error_message: str) -> None:
        """Set error message."""
        with self._lock:
            self._error_message = error_message
            self._last_update_time = datetime.utcnow()
    
    def get_error(self) -> Optional[str]:
        """Get error message."""
        with self._lock:
            return self._error_message
    
    def get_last_update_time(self) -> Optional[datetime]:
        """Get time of last update (for detecting changes)."""
        with self._lock:
            return self._last_update_time
    
    def reset(self) -> None:
        """Reset all state to initial values."""
        with self._lock:
            self._status = "idle"
            self._current_stage = None
            self._progress = 0
            self._elapsed_time = 0.0
            self._start_time = None
            self._stage_results = {}
            self._stage_statuses = {}
            self._events = []
            self._final_result = None
            self._error_message = None
            self._execution_state_id = None
            self._last_update_time = None
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the entire state (for debugging)."""
        with self._lock:
            return {
                "status": self._status,
                "current_stage": self._current_stage,
                "progress": self._progress,
                "elapsed_time": self._elapsed_time,
                "start_time": self._start_time.isoformat() if self._start_time else None,
                    "stage_results_keys": list(self._stage_results.keys()),
                "stage_statuses": dict(self._stage_statuses),
                "events_count": len(self._events),
                "execution_state_id": self._execution_state_id,
                "error": self._error_message,
                "has_final_result": self._final_result is not None,
            }

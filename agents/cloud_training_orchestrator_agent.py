from __future__ import annotations

import ctypes
import datetime
import json
import os
import platform
import subprocess
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple


class CloudTrainingOrchestratorAgent:
    """Agent for orchestrating cloud and local training jobs."""

    DEFAULT_JOBS_FILE = "training_jobs.json"
    JOB_STATUSES = {"Queued", "Running", "Completed", "Failed", "Cancelled"}
    HARDWARE_COSTS = {
        "Local CPU": 0.0,
        "Cloud CPU": 0.5,
        "GPU": 2.0,
        "Multi GPU": 5.0,
        "Distributed Cluster": 10.0,
    }

    def __init__(self, jobs_path: str = DEFAULT_JOBS_FILE) -> None:
        self.jobs_path = jobs_path
        self._ensure_jobs_file()

    def _ensure_jobs_file(self) -> None:
        if not os.path.exists(self.jobs_path):
            self._save_store({
                "jobs": [],
                "resource_snapshots": [],
                "optimization_reports": [],
            })

    def _load_store(self) -> Dict[str, Any]:
        try:
            with open(self.jobs_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"jobs": [], "resource_snapshots": [], "optimization_reports": []}

    def _save_store(self, data: Dict[str, Any]) -> None:
        with open(self.jobs_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)

    def _now(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def _generate_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex}"

    def detect_system_resources(self) -> Dict[str, Any]:
        cpu_count = os.cpu_count() or 1
        memory = self._get_memory_info()
        gpu = self._detect_gpu_info()
        environment = {
            "os": platform.platform(),
            "python_version": platform.python_version(),
        }
        snapshot = {
            "cpu": {"count": cpu_count},
            "memory": memory,
            "gpu": gpu,
            "environment": environment,
        }
        store = self._load_store()
        store["resource_snapshots"].append({"timestamp": self._now(), "snapshot": snapshot})
        store["resource_snapshots"] = store["resource_snapshots"][-10000:]
        self._save_store(store)
        return snapshot

    def _get_memory_info(self) -> Dict[str, Any]:
        total_bytes, available_bytes = None, None
        system = platform.system()
        try:
            if system == "Windows":
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                total_bytes = stat.ullTotalPhys
                available_bytes = stat.ullAvailPhys
            elif system == "Linux":
                total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                available_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
            elif system == "Darwin":
                total_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
                vm_stat = subprocess.check_output(["vm_stat"]).decode("utf-8")
                free_pages = 0
                page_size = 4096
                for line in vm_stat.splitlines():
                    if "Pages free" in line or "Pages inactive" in line:
                        free_pages += int(line.split(":")[1].strip().strip("."))
                available_bytes = free_pages * page_size
        except Exception:
            pass

        return {
            "total_bytes": int(total_bytes) if total_bytes is not None else None,
            "available_bytes": int(available_bytes) if available_bytes is not None else None,
            "total_gb": round(total_bytes / 1024**3, 2) if total_bytes else None,
            "available_gb": round(available_bytes / 1024**3, 2) if available_bytes else None,
        }

    def _detect_gpu_info(self) -> Dict[str, Any]:
        gpu_info = {"available": False, "count": 0, "memory_total_gb": None, "details": []}
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                gpu_info["available"] = True
                gpu_info["count"] = len(lines)
                total_memory = 0.0
                for line in lines:
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) >= 2:
                        name, memory_str = parts[0], parts[1]
                        try:
                            mem_gb = float(memory_str) / 1024.0
                        except ValueError:
                            mem_gb = 0.0
                        total_memory += mem_gb
                        gpu_info["details"].append({"name": name, "memory_gb": round(mem_gb, 2)})
                gpu_info["memory_total_gb"] = round(total_memory, 2)
        except Exception:
            pass
        return gpu_info

    def recommend_training_strategy(self, dataset_metadata: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        rows = int(dataset_metadata.get("row_count", 0))
        columns = int(dataset_metadata.get("column_count", 0))
        model_type = model_type.lower()

        if rows <= 10000:
            size_label = "Small"
            infrastructure = "Local CPU"
            estimated_hours = max(0.5, rows / 20000.0)
            expected_cost = self.estimate_training_cost(estimated_hours, infrastructure)["estimated_cost"]
        elif rows <= 100000:
            size_label = "Medium"
            infrastructure = "Cloud CPU"
            estimated_hours = max(1.0, rows / 50000.0)
            expected_cost = self.estimate_training_cost(estimated_hours, infrastructure)["estimated_cost"]
        elif rows <= 1000000:
            size_label = "Large"
            infrastructure = "GPU"
            estimated_hours = max(2.0, rows / 200000.0)
            expected_cost = self.estimate_training_cost(estimated_hours, infrastructure)["estimated_cost"]
        else:
            size_label = "Massive"
            infrastructure = "Distributed Cluster"
            estimated_hours = max(4.0, rows / 500000.0)
            expected_cost = self.estimate_training_cost(estimated_hours, infrastructure)["estimated_cost"]

        if "deep" in model_type or "transformer" in model_type or "cnn" in model_type:
            infrastructure = "GPU" if size_label != "Massive" else "Distributed Cluster"

        reasoning = (
            f"Dataset size is {size_label} ({rows} rows, {columns} columns). "
            f"Selected {infrastructure} based on model type '{model_type}' and expected scalability needs."
        )

        return {
            "recommended_infrastructure": infrastructure,
            "dataset_size": size_label,
            "estimated_training_time_hours": round(estimated_hours, 2),
            "expected_cost": round(expected_cost, 2),
            "currency": "USD",
            "reasoning": reasoning,
        }

    def estimate_training_cost(self, hours: float, hardware: str) -> Dict[str, Any]:
        hardware_key = hardware if hardware in self.HARDWARE_COSTS else "Cloud CPU"
        rate = self.HARDWARE_COSTS.get(hardware_key, 1.0)
        estimated_cost = float(hours) * rate
        breakdown = {
            "hardware_type": hardware_key,
            "hourly_rate": rate,
            "hours": float(round(hours, 2)),
        }
        return {"estimated_cost": round(estimated_cost, 2), "currency": "USD", "breakdown": breakdown}

    def create_training_job(
        self,
        project_id: str,
        model_name: str,
        dataset_info: Dict[str, Any],
        strategy: Dict[str, Any],
    ) -> Dict[str, Any]:
        store = self._load_store()
        job_id = self._generate_id("job")
        job = {
            "job_id": job_id,
            "project_id": project_id,
            "model_name": model_name,
            "dataset_info": dataset_info,
            "strategy": strategy,
            "status": "Queued",
            "created_at": self._now(),
            "start_time": None,
            "end_time": None,
            "logs": [],
        }
        store["jobs"].append(job)
        store["jobs"] = store["jobs"][-10000:]
        self._save_store(store)
        return job

    def _update_job(self, job_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
        store = self._load_store()
        job = next((item for item in store.get("jobs", []) if item.get("job_id") == job_id), None)
        if not job:
            return None
        for key, value in updates.items():
            if key in job:
                job[key] = value
        self._save_store(store)
        return job

    def start_job(self, job_id: str, log_message: Optional[str] = None) -> Optional[Dict[str, Any]]:
        job = self._update_job(job_id, status="Running", start_time=self._now())
        if job is None:
            return None
        if log_message:
            job["logs"].append({"timestamp": self._now(), "message": log_message})
            self._update_job(job_id, logs=job["logs"])
        return job

    def complete_job(self, job_id: str, log_message: Optional[str] = None) -> Optional[Dict[str, Any]]:
        job = self._update_job(job_id, status="Completed", end_time=self._now())
        if job is None:
            return None
        if log_message:
            job["logs"].append({"timestamp": self._now(), "message": log_message})
            self._update_job(job_id, logs=job["logs"])
        return job

    def fail_job(self, job_id: str, error_message: str) -> Optional[Dict[str, Any]]:
        job = self._update_job(job_id, status="Failed", end_time=self._now())
        if job is None:
            return None
        job["logs"].append({"timestamp": self._now(), "message": f"Failed: {error_message}"})
        self._update_job(job_id, logs=job["logs"])
        return job

    def cancel_job(self, job_id: str, reason: Optional[str] = None) -> Optional[Dict[str, Any]]:
        job = self._update_job(job_id, status="Cancelled", end_time=self._now())
        if job is None:
            return None
        if reason:
            job["logs"].append({"timestamp": self._now(), "message": f"Cancelled: {reason}"})
            self._update_job(job_id, logs=job["logs"])
        return job

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        store = self._load_store()
        return next((item for item in store.get("jobs", []) if item.get("job_id") == job_id), None)

    def optimize_resource_usage(self) -> Dict[str, Any]:
        resources = self.detect_system_resources()
        suggestions: List[str] = []
        gpu_available = resources["gpu"].get("available", False)
        memory_gb = resources["memory"].get("available_gb")
        cpu_count = resources["cpu"].get("count", 1)

        if gpu_available:
            suggestions.append("Enable GPU acceleration for deep learning and large model training.")
        else:
            suggestions.append("Use parallel CPU training and batch processing to conserve memory.")

        if memory_gb is not None and memory_gb < 16:
            suggestions.append("Reduce batch size or use streaming data ingestion to reduce memory usage.")
        if cpu_count >= 8:
            suggestions.append("Use multi-core parallelism for model training and data preprocessing.")
        if memory_gb is not None and memory_gb >= 32 and not gpu_available:
            suggestions.append("Consider cloud GPU instances if training is compute-bound.")

        report = {
            "generated_at": self._now(),
            "suggestions": suggestions,
            "resource_snapshot": resources,
        }
        store = self._load_store()
        store["optimization_reports"].append(report)
        store["optimization_reports"] = store["optimization_reports"][-10000:]
        self._save_store(store)
        return report

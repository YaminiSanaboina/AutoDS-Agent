from __future__ import annotations

import datetime
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple


class AgentMemoryDatabase:
    """Central memory system for all agents to store, retrieve, and learn from shared knowledge."""

    DEFAULT_MEMORY_FILE = "storage/memory/agent_memory.json"
    MAX_MEMORIES = 100000

    CATEGORIES = {
        "dataset_analysis",
        "model_training",
        "failure_analysis",
        "experiment_result",
        "deployment_insight",
        "performance_tuning",
        "best_practice",
        "cross_agent_learning",
    }

    def __init__(self, memory_path: Optional[str] = None):
        self.memory_path = memory_path or self.DEFAULT_MEMORY_FILE
        self._ensure_storage()
        self._load_memories()

    def _now(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def _ensure_storage(self) -> None:
        if not os.path.exists(self.memory_path):
            initial = {"memories": []}
            with open(self.memory_path, "w", encoding="utf-8") as fh:
                json.dump(initial, fh, indent=2)

    def _load_memories(self) -> None:
        try:
            with open(self.memory_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                self.memories = data.get("memories", [])
        except Exception:
            self.memories = []

    def _save_memories(self) -> None:
        with open(self.memory_path, "w", encoding="utf-8") as fh:
            json.dump({"memories": self.memories}, fh, indent=2)
        self._cleanup_if_needed()

    def add_memory(
        self,
        agent: str,
        category: str,
        title: str,
        content: str,
        tags: List[str],
        importance: float = 0.5,
        success_score: float = 0.5,
    ) -> Dict[str, Any]:
        """Add a new memory entry."""
        memory_id = f"MEM_{uuid.uuid4().hex[:8]}"
        if category not in self.CATEGORIES:
            category = "best_practice"
        importance = max(0.0, min(1.0, importance))
        success_score = max(0.0, min(1.0, success_score))

        entry = {
            "memory_id": memory_id,
            "timestamp": self._now(),
            "agent": agent,
            "category": category,
            "title": title,
            "content": content,
            "tags": list(set(tags)),
            "importance": importance,
            "success_score": success_score,
        }
        self.memories.append(entry)
        self._save_memories()
        return entry

    def _time_decay_factor(self, timestamp_iso: str) -> float:
        """Compute age decay factor (0.8 to 1.0)."""
        try:
            ts = datetime.datetime.fromisoformat(timestamp_iso)
            now = datetime.datetime.now(datetime.timezone.utc)
            age_days = (now - ts).days
            # reduce by 0.002 per day (~20% per year)
            decay = max(0.8, 1.0 - (age_days * 0.002))
            return decay
        except Exception:
            return 1.0

    def search_memory(self, query: str, tags: Optional[List[str]] = None, agent: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search memories by keywords, tags, agent and return ranked results."""
        query_lower = query.lower()
        results = []

        for mem in self.memories:
            score = 0.0
            # keyword match
            title_match = sum(1 for w in query_lower.split() if w in mem.get("title", "").lower())
            content_match = sum(1 for w in query_lower.split() if w in mem.get("content", "").lower())
            score += title_match * 0.3 + content_match * 0.1

            # tag match
            if tags:
                tag_match = len(set(tags) & set(mem.get("tags", [])))
                score += tag_match * 0.2

            # agent filter
            if agent and mem.get("agent") != agent:
                continue

            # importance and success
            score += mem.get("importance", 0.5) * 0.2
            score += mem.get("success_score", 0.5) * 0.2

            # age decay
            score *= self._time_decay_factor(mem.get("timestamp", self._now()))

            if score > 0:
                results.append((score, mem))

        # sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in results[:top_k]]

    def get_agent_history(self, agent_name: str) -> List[Dict[str, Any]]:
        """Retrieve all memories from a specific agent."""
        return [m for m in self.memories if m.get("agent") == agent_name]

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        original_len = len(self.memories)
        self.memories = [m for m in self.memories if m.get("memory_id") != memory_id]
        if len(self.memories) < original_len:
            self._save_memories()
            return True
        return False

    def retrieve_relevant_knowledge(self, task_description: str, top_k: int = 5) -> List[str]:
        """Retrieve top relevant memories and extract key learnings."""
        results = self.search_memory(task_description, top_k=top_k)
        learnings = []
        for mem in results:
            # extract key insights
            content = mem.get("content", "")
            if len(content) > 50:
                learnings.append(content[:200])
            else:
                learnings.append(content)
        return learnings

    def share_learning(self, source_agent: str, target_agent: str, learning: str) -> Dict[str, Any]:
        """Record cross-agent learning."""
        return self.add_memory(
            agent=target_agent,
            category="cross_agent_learning",
            title=f"Learning from {source_agent}",
            content=learning,
            tags=[source_agent, target_agent],
            importance=0.8,
            success_score=0.7,
        )

    def learn_from_experiment(self, experiment_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract and store learnings from experiment results."""
        if not experiment_data:
            return None
        return self.add_memory(
            agent=experiment_data.get("agent", "ExperimentMemoryAgent"),
            category="experiment_result",
            title=f"Experiment: {experiment_data.get('title', 'Untitled')}",
            content=f"Metric: {experiment_data.get('result', {}).get('best_metric', 'unknown')} at {experiment_data.get('result', {}).get('best_params', {})}",
            tags=experiment_data.get("tags", []),
            importance=experiment_data.get("importance", 0.6),
            success_score=experiment_data.get("success_score", 0.5),
        )

    def learn_from_failure(self, failure_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract and store learnings from failures."""
        if not failure_data:
            return None
        return self.add_memory(
            agent=failure_data.get("agent", "SelfHealingAgent"),
            category="failure_analysis",
            title=f"Failure: {failure_data.get('title', 'Unknown')}",
            content=f"Root cause: {failure_data.get('root_cause', 'unknown')}. Solution: {failure_data.get('solution', 'pending')}",
            tags=failure_data.get("tags", ["failure"]),
            importance=failure_data.get("importance", 0.7),
            success_score=0.3,
        )

    def learn_from_project(self, project_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract and store learnings from project outcomes."""
        if not project_data:
            return None
        return self.add_memory(
            agent=project_data.get("agent", "ProjectWorkspaceAgent"),
            category="deployment_insight",
            title=f"Project: {project_data.get('title', 'Unknown')}",
            content=f"Outcome: {project_data.get('outcome', 'unknown')}. Performance: {project_data.get('performance', 'not measured')}",
            tags=project_data.get("tags", []),
            importance=project_data.get("importance", 0.6),
            success_score=project_data.get("success_score", 0.5),
        )

    def recommend_best_practices(self, domain: str, problem_type: str) -> List[str]:
        """Provide best practice recommendations for a domain/problem."""
        query = f"{domain} {problem_type}"
        relevant = self.search_memory(query, top_k=10)
        recommendations = set()
        for mem in relevant:
            if mem.get("success_score", 0) > 0.7:
                content = mem.get("content", "")
                if content:
                    recommendations.add(content)
        # add generic recommendations
        if "classification" in problem_type.lower():
            recommendations.add("Validate fairness and bias in predictions.")
            recommendations.add("Use cross-validation to assess model stability.")
        if domain.lower() == "healthcare":
            recommendations.add("Prioritize recall and explainability.")
            recommendations.add("Monitor drift after deployment.")
        return list(recommendations)[:5]

    def generate_memory_report(self) -> Dict[str, Any]:
        """Generate analytics report on memory database."""
        agent_counts = {}
        for mem in self.memories:
            agent = mem.get("agent")
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        top_agents = sorted(agent_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        category_counts = {}
        for mem in self.memories:
            cat = mem.get("category")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        avg_success = sum(m.get("success_score", 0) for m in self.memories) / max(1, len(self.memories))
        # health score based on success and recency
        recent_count = sum(1 for m in self.memories if self._time_decay_factor(m.get("timestamp")) > 0.95)
        health = min(100, int(avg_success * 100 + (recent_count / max(1, len(self.memories))) * 10))

        return {
            "total_memories": len(self.memories),
            "top_agents": [ag for ag, _ in top_agents],
            "categories": category_counts,
            "average_success": round(avg_success, 2),
            "knowledge_health_score": health,
        }

    def _cleanup_if_needed(self) -> None:
        """Remove lowest-scoring memories if exceeding limit."""
        if len(self.memories) <= self.MAX_MEMORIES:
            return

        # score each memory
        scored = []
        for mem in self.memories:
            score = (
                mem.get("importance", 0.5)
                * mem.get("success_score", 0.5)
                * self._time_decay_factor(mem.get("timestamp"))
            )
            scored.append((score, mem))

        # keep top MAX_MEMORIES
        scored.sort(key=lambda x: x[0], reverse=True)
        self.memories = [mem for _, mem in scored[:self.MAX_MEMORIES]]
        with open(self.memory_path, "w", encoding="utf-8") as fh:
            json.dump({"memories": self.memories}, fh, indent=2)

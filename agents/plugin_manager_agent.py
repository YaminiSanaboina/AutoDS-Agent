from __future__ import annotations

import importlib.util
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional


class PluginManagerAgent:
    """Manages discovery, validation, loading and execution of external agent plugins.

    Plugins are expected under the `plugins/` folder. Each plugin must have:
      - manifest.json
      - agent.py

    The manifest must include: name, version, description, author, entry_class
    """

    DEFAULT_PLUGINS_DIR = "plugins"
    DEFAULT_REGISTRY = "plugin_registry.json"

    REQUIRED_MANIFEST_FIELDS = {"name", "version", "description", "author", "entry_class"}
    ALLOWED_CAPABILITIES = {"read_dataset", "analyze_data", "generate_report"}
    FORBIDDEN_CAPABILITIES = {"system", "delete_files", "network", "env_modify"}

    def __init__(self, plugins_dir: Optional[str] = None, registry_path: Optional[str] = None):
        self.plugins_dir = plugins_dir or self.DEFAULT_PLUGINS_DIR
        self.registry_path = registry_path or self.DEFAULT_REGISTRY
        self.loaded_plugins: Dict[str, Any] = {}
        self._ensure_registry()
        self._load_registry()

    def _ensure_registry(self) -> None:
        if not os.path.exists(self.registry_path):
            initial = {"plugins": []}
            with open(self.registry_path, "w", encoding="utf-8") as fh:
                json.dump(initial, fh, indent=2)

    def _load_registry(self) -> None:
        try:
            with open(self.registry_path, "r", encoding="utf-8") as fh:
                self.registry = json.load(fh)
        except Exception:
            self.registry = {"plugins": []}

    def _save_registry(self) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as fh:
            json.dump(self.registry, fh, indent=2)

    def discover_plugins(self) -> List[Dict[str, Any]]:
        """Scan the plugins directory and return basic metadata for each plugin folder."""
        found = []
        if not os.path.isdir(self.plugins_dir):
            return found

        for name in sorted(os.listdir(self.plugins_dir)):
            p = os.path.join(self.plugins_dir, name)
            if not os.path.isdir(p):
                continue
            manifest_path = os.path.join(p, "manifest.json")
            meta = {"name": name, "status": "available", "path": p}
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as fh:
                        m = json.load(fh)
                        meta.update({
                            "name": m.get("name", name),
                            "version": m.get("version"),
                            "author": m.get("author"),
                            "description": m.get("description"),
                        })
                except Exception:
                    meta["status"] = "invalid_manifest"
            found.append(meta)
        return found

    def validate_plugin(self, plugin_path: str) -> Dict[str, Any]:
        """Validate plugin structure and manifest contents."""
        errors: List[str] = []
        warnings: List[str] = []

        if not os.path.isdir(plugin_path):
            errors.append("plugin_path is not a directory")
            return {"valid": False, "errors": errors, "warnings": warnings}

        manifest_file = os.path.join(plugin_path, "manifest.json")
        agent_file = os.path.join(plugin_path, "agent.py")

        if not os.path.exists(manifest_file):
            errors.append("missing manifest.json")
        if not os.path.exists(agent_file):
            errors.append("missing agent.py")

        manifest = {}
        if os.path.exists(manifest_file):
            try:
                with open(manifest_file, "r", encoding="utf-8") as fh:
                    manifest = json.load(fh)
            except Exception as exc:
                errors.append(f"manifest parse error: {exc}")

        # required fields
        for field in self.REQUIRED_MANIFEST_FIELDS:
            if field not in manifest:
                errors.append(f"manifest missing field: {field}")

        # version basic check
        version = manifest.get("version")
        if version:
            parts = str(version).split(".")
            if not all(p.isdigit() for p in parts):
                errors.append("invalid version format")

        # duplicate name
        pname = manifest.get("name")
        if pname:
            for p in self.registry.get("plugins", []):
                if p.get("name") == pname:
                    warnings.append("plugin name duplicates installed plugin")

        # capabilities checks
        caps = manifest.get("capabilities", []) or []
        for c in caps:
            if c in self.FORBIDDEN_CAPABILITIES:
                errors.append(f"forbidden capability requested: {c}")
            if c not in self.ALLOWED_CAPABILITIES:
                warnings.append(f"unknown capability: {c}")

        # basic python file sanity
        if os.path.exists(agent_file):
            try:
                with open(agent_file, "r", encoding="utf-8") as fh:
                    src = fh.read()
                    # avoid executing; ensure contains class name
                    entry = manifest.get("entry_class")
                    if entry and entry not in src:
                        warnings.append("entry_class not obviously present in agent.py source")
            except Exception as exc:
                errors.append(f"agent.py read error: {exc}")

        valid = not errors
        return {"valid": valid, "errors": errors, "warnings": warnings, "manifest": manifest}

    def load_plugin(self, plugin_name: str) -> Dict[str, Any]:
        """Load plugin by folder name from plugins_dir and instantiate its entry class."""
        plugin_path = os.path.join(self.plugins_dir, plugin_name)
        if not os.path.isdir(plugin_path):
            raise FileNotFoundError("plugin not found")

        manifest_path = os.path.join(plugin_path, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError("manifest missing")

        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)

        entry_class = manifest.get("entry_class")
        if not entry_class:
            raise ValueError("manifest missing entry_class")

        agent_file = os.path.join(plugin_path, "agent.py")
        spec_name = f"plugin_{plugin_name}"
        try:
            spec = importlib.util.spec_from_file_location(spec_name, agent_file)
            module = importlib.util.module_from_spec(spec)
            loader = spec.loader
            assert loader is not None
            loader.exec_module(module)
        except Exception as exc:
            # record load failure
            self._record_health(manifest.get("name") or plugin_name, load_failure=True)
            raise

        if not hasattr(module, entry_class):
            raise ImportError(f"entry class {entry_class} not found in module")

        cls = getattr(module, entry_class)
        try:
            instance = cls()
        except Exception as exc:
            self._record_health(manifest.get("name") or plugin_name, load_failure=True)
            raise

        self.loaded_plugins[manifest.get("name") or plugin_name] = {"instance": instance, "manifest": manifest, "path": plugin_path}

        # update registry
        self._register_plugin(manifest, plugin_path)
        return {"name": manifest.get("name"), "loaded": True}

    def _register_plugin(self, manifest: Dict[str, Any], path: str) -> None:
        name = manifest.get("name")
        now = time.time()
        plugins = self.registry.setdefault("plugins", [])
        existing = next((p for p in plugins if p.get("name") == name), None)
        if existing:
            existing.update({"version": manifest.get("version"), "path": path, "installed_at": now})
        else:
            plugins.append({
                "name": name,
                "version": manifest.get("version"),
                "enabled": False,
                "path": path,
                "installed_at": now,
                "health": {"load_failures": 0, "execution_failures": 0, "execution_count": 0, "last_used": None},
            })
        self._save_registry()

    def enable_plugin(self, name: str) -> bool:
        for p in self.registry.get("plugins", []):
            if p.get("name") == name:
                p["enabled"] = True
                self._save_registry()
                return True
        return False

    def disable_plugin(self, name: str) -> bool:
        for p in self.registry.get("plugins", []):
            if p.get("name") == name:
                p["enabled"] = False
                self._save_registry()
                return True
        return False

    def execute_plugin(self, plugin_name: str, input_data: Any) -> Dict[str, Any]:
        record = self.loaded_plugins.get(plugin_name)
        if not record:
            raise ValueError("plugin not loaded")

        manifest = record.get("manifest", {})
        # security: check capabilities
        caps = set(manifest.get("capabilities", []) or [])
        if caps & self.FORBIDDEN_CAPABILITIES:
            return {"success": False, "error": "plugin requests forbidden capabilities"}

        # check enabled
        reg = next((p for p in self.registry.get("plugins", []) if p.get("name") == plugin_name), None)
        if reg is None or not reg.get("enabled"):
            return {"success": False, "error": "plugin not enabled"}

        instance = record.get("instance")
        start = time.time()
        try:
            result = instance.run(input_data)
            duration = time.time() - start
            # record health
            self._record_health(plugin_name, execution=True)
            return {"success": True, "result": result, "duration": duration}
        except Exception as exc:
            self._record_health(plugin_name, execution_failure=True)
            return {"success": False, "error": str(exc), "trace": traceback.format_exc()}

    def _record_health(self, plugin_name: str, load_failure: bool = False, execution_failure: bool = False, execution: bool = False) -> None:
        p = next((p for p in self.registry.get("plugins", []) if p.get("name") == plugin_name), None)
        if not p:
            return
        health = p.setdefault("health", {"load_failures": 0, "execution_failures": 0, "execution_count": 0, "last_used": None})
        if load_failure:
            health["load_failures"] = health.get("load_failures", 0) + 1
        if execution_failure:
            health["execution_failures"] = health.get("execution_failures", 0) + 1
        if execution:
            health["execution_count"] = health.get("execution_count", 0) + 1
            health["last_used"] = time.time()
        self._save_registry()

    def get_marketplace_catalog(self) -> List[Dict[str, Any]]:
        items = []
        discovered = self.discover_plugins()
        for d in discovered:
            manifest_path = os.path.join(d.get("path"), "manifest.json")
            entry = {"name": d.get("name"), "category": None, "rating": None}
            try:
                if os.path.exists(manifest_path):
                    with open(manifest_path, "r", encoding="utf-8") as fh:
                        m = json.load(fh)
                        entry["category"] = m.get("category")
                        entry["rating"] = m.get("rating")
            except Exception:
                pass
            items.append(entry)
        return items

    def get_plugin_health(self, plugin_name: Optional[str] = None) -> Any:
        if plugin_name:
            p = next((p for p in self.registry.get("plugins", []) if p.get("name") == plugin_name), None)
            return p.get("health") if p else None
        return {p.get("name"): p.get("health") for p in self.registry.get("plugins", [])}

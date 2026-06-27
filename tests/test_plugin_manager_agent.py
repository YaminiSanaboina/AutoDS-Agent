import json
import os
import shutil
import time
from pathlib import Path

import pytest

from agents.plugin_manager_agent import PluginManagerAgent


def make_plugin(tmpdir, name, manifest_data, agent_source):
    p = Path(tmpdir) / name
    p.mkdir(parents=True, exist_ok=True)
    (p / "manifest.json").write_text(json.dumps(manifest_data))
    (p / "agent.py").write_text(agent_source)
    return str(p)


def test_discover_and_validate_and_load(tmp_path, monkeypatch):
    base = tmp_path / "plugins"
    base.mkdir()

    manifest = {
        "name": "TestAgent",
        "version": "1.0.0",
        "description": "A test plugin",
        "author": "Dev",
        "entry_class": "TestAgent",
        "capabilities": ["read_dataset"]
    }
    agent_src = """
class TestAgent:
    def run(self, data):
        return {"echo": data}
"""
    make_plugin(base, "test_agent", manifest, agent_src)

    registry = str(tmp_path / "plugin_registry.json")
    pm = PluginManagerAgent(plugins_dir=str(base), registry_path=registry)

    discovered = pm.discover_plugins()
    assert any(d["name"] == "test_agent" or d.get("name") == "TestAgent" for d in discovered)

    # validate
    plugin_path = os.path.join(str(base), "test_agent")
    valid = pm.validate_plugin(plugin_path)
    assert valid["valid"] is True

    # load
    res = pm.load_plugin("test_agent")
    assert res["loaded"] is True

    # enable and execute
    assert pm.enable_plugin("TestAgent") or pm.enable_plugin("test_agent")
    # plugin registry uses manifest name
    name = manifest["name"]
    out = pm.execute_plugin(name, {"a": 1})
    assert out["success"] is True
    assert out["result"]["echo"]["a"] == 1

    # health should show execution count
    health = pm.get_plugin_health(name)
    assert health is not None


def test_manifest_missing_fields(tmp_path):
    base = tmp_path / "plugins2"
    base.mkdir()
    manifest = {"name": "BadAgent", "version": "1.0"}
    agent_src = "class BadAgent: pass"
    p = make_plugin(base, "bad_agent", manifest, agent_src)

    pm = PluginManagerAgent(plugins_dir=str(base), registry_path=str(tmp_path / "reg2.json"))
    v = pm.validate_plugin(p)
    assert not v["valid"]
    assert any("missing field" in e or "manifest missing field" in e for e in v["errors"]) or v["errors"]


def test_forbidden_capability_blocked(tmp_path):
    base = tmp_path / "plugins3"
    base.mkdir()
    manifest = {
        "name": "EvilAgent",
        "version": "1.0.0",
        "description": "bad",
        "author": "x",
        "entry_class": "EvilAgent",
        "capabilities": ["network"]
    }
    agent_src = "class EvilAgent:\n    def run(self,data):\n        return {'ok':True}"
    p = make_plugin(base, "evil", manifest, agent_src)

    pm = PluginManagerAgent(plugins_dir=str(base), registry_path=str(tmp_path / "reg3.json"))
    v = pm.validate_plugin(p)
    assert not v["valid"]


def test_enable_disable_registry_updates(tmp_path):
    base = tmp_path / "plugins4"
    base.mkdir()
    manifest = {
        "name": "ToggleAgent",
        "version": "1.0.0",
        "description": "toggle",
        "author": "me",
        "entry_class": "ToggleAgent",
    }
    agent_src = "class ToggleAgent:\n    def run(self,data):\n        return {'ok':True}"
    make_plugin(base, "toggle", manifest, agent_src)

    reg = str(tmp_path / "reg4.json")
    pm = PluginManagerAgent(plugins_dir=str(base), registry_path=reg)
    pm.load_plugin("toggle")
    assert pm.enable_plugin("ToggleAgent")
    # read registry
    with open(reg, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    p = next((x for x in data["plugins"] if x["name"] == "ToggleAgent"), None)
    assert p and p["enabled"] is True
    assert pm.disable_plugin("ToggleAgent")
    with open(reg, "r", encoding="utf-8") as fh:
        data2 = json.load(fh)
    p2 = next((x for x in data2["plugins"] if x["name"] == "ToggleAgent"), None)
    assert p2 and p2["enabled"] is False

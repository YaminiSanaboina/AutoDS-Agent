"""One-off repository reference analyzer for unused-file audit."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    "venv",
    "__pycache__",
    ".git",
    "deployment_package",
    "node_modules",
    ".pytest_cache",
    "htmlcov",
    "artifacts",
    "model_artifacts",
    "reports",
    "outputs",
    "logs",
    "data",
    "datasets",
    "models",
    "registry",
    "memory",
    "experiments",
    "notebooks",
    "tmp",
    "cache",
    "smart_mode_cache",
    "uploads",
    "storage",
    "workspace",
    "project_workspace",
    "agent_memory",
    "knowledge_base",
    "training_jobs",
    "agent_events",
}


def should_skip(path: Path) -> bool:
    return bool(set(path.parts) & SKIP_DIRS) or path.suffix != ".py"


def module_id(rel: str) -> str:
    return rel.replace("/", ".").replace(".py", "")


def collect_py_files() -> list[str]:
    files: list[str] = []
    for path in ROOT.rglob("*.py"):
        if should_skip(path):
            continue
        files.append(path.relative_to(ROOT).as_posix())
    return sorted(files)


def search_patterns(rel: str) -> list[str]:
    mod = module_id(rel)
    stem = Path(rel).stem
    patterns = {
        mod,
        stem,
        f"from {mod} import",
        f"import {mod}",
    }
    parts = mod.split(".")
    if len(parts) > 1:
        patterns.add(".".join(parts[-2:]))
        patterns.add(parts[-1])
    if rel.startswith("ui/legacy_pages/"):
        patterns.add(f"legacy_pages.{stem}")
        patterns.add(f"ui.legacy_pages.{stem}")
    if rel.startswith("ui/pages/"):
        patterns.add(f"pages.{stem}")
        patterns.add(f"ui.pages.{stem}")
    return sorted(patterns)


def main() -> None:
    py_files = collect_py_files()
    contents: dict[str, str] = {}
    for rel in py_files:
        contents[rel] = (ROOT / rel).read_text(encoding="utf-8", errors="ignore")

    refs: dict[str, list[str]] = defaultdict(list)
    test_refs: dict[str, list[str]] = defaultdict(list)

    for target in py_files:
        patterns = search_patterns(target)
        for rel, text in contents.items():
            if rel == target:
                continue
            matched = False
            for pat in patterns:
                if pat in text:
                    matched = True
                    break
                if re.search(rf"\bfrom\s+{re.escape(pat)}\s+import\b", text):
                    matched = True
                    break
                if re.search(rf"\bimport\s+{re.escape(pat)}\b", text):
                    matched = True
                    break
            if matched:
                if rel.startswith("tests/"):
                    test_refs[target].append(rel)
                else:
                    refs[target].append(rel)

    for mapping in (refs, test_refs):
        for key in mapping:
            mapping[key] = sorted(set(mapping[key]))

    categories = {
        "agents": [],
        "ui": [],
        "utils": [],
        "api": [],
        "scripts": [],
        "tools": [],
        "tests": [],
        "root": [],
    }
    for rel in py_files:
        if rel.startswith("agents/"):
            categories["agents"].append(rel)
        elif rel.startswith("ui/"):
            categories["ui"].append(rel)
        elif rel.startswith("utils/"):
            categories["utils"].append(rel)
        elif rel.startswith("api/"):
            categories["api"].append(rel)
        elif rel.startswith("scripts/"):
            categories["scripts"].append(rel)
        elif rel.startswith("tools/"):
            categories["tools"].append(rel)
        elif rel.startswith("tests/"):
            categories["tests"].append(rel)
        else:
            categories["root"].append(rel)

    print("REPO FILE AUDIT")
    print("=" * 80)
    for cat, files in categories.items():
        print(f"\n## {cat.upper()} ({len(files)} files)")
        for rel in files:
            runtime = refs.get(rel, [])
            tests = test_refs.get(rel, [])
            all_refs = sorted(set(runtime + tests))
            status = "REFERENCED" if all_refs else "NO_REFS"
            print(f"{status:12} {rel}")
            if all_refs:
                print(f"             refs: {', '.join(all_refs[:8])}{' ...' if len(all_refs) > 8 else ''}")

    zero_refs = [rel for rel in py_files if not refs.get(rel) and not test_refs.get(rel)]
    print("\n\nZERO REFERENCES ANYWHERE:")
    for rel in zero_refs:
        print(rel)
    print(f"Count: {len(zero_refs)}")


if __name__ == "__main__":
    main()

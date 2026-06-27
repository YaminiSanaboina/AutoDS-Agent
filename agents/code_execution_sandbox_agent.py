import ast
import datetime
import io
import json
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from agents.experiment_memory_agent import ExperimentMemoryAgent
from agents.self_healing_agent import SelfHealingAgent


class UnsafeCodeError(Exception):
    pass


class UnsafeCodeDetector(ast.NodeVisitor):
    ALLOWED_MODULES = {
        "pandas",
        "numpy",
        "sklearn",
        "matplotlib",
        "seaborn",
        "math",
        "statistics",
    }
    BANNED_MODULES = {
        "os",
        "subprocess",
        "socket",
        "shutil",
        "pathlib",
        "sys",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "paramiko",
    }
    BANNED_FUNCTIONS = {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
    }
    BANNED_ATTRS = {
        ("os", "remove"),
        ("os", "system"),
        ("subprocess", "run"),
        ("subprocess", "Popen"),
        ("socket", "connect"),
        ("socket", "socket"),
        ("shutil", "rmtree"),
        ("shutil", "remove"),
        ("pathlib", "Path"),
    }

    def __init__(self) -> None:
        self.issues: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module_name = alias.name.split(".")[0]
            if module_name in self.BANNED_MODULES or module_name not in self.ALLOWED_MODULES:
                self.issues.append(f"Import of module '{alias.name}' is not permitted.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            self.issues.append("Relative imports are not permitted.")
            return
        module_name = node.module.split(".")[0]
        if module_name in self.BANNED_MODULES or module_name not in self.ALLOWED_MODULES:
            self.issues.append(f"Import from module '{node.module}' is not permitted.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.BANNED_FUNCTIONS:
                self.issues.append(f"Use of function '{func_name}' is not permitted.")
        elif isinstance(node.func, ast.Attribute):
            parts = self._flatten_attribute(node.func)
            if parts and tuple(parts[:2]) in self.BANNED_ATTRS:
                self.issues.append(f"Use of '{'.'.join(parts[:2])}' is not permitted.")
            if parts and parts[0] in self.BANNED_MODULES:
                self.issues.append(f"Access to banned module '{parts[0]}' is not permitted.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name):
            name = node.value.id
            if name in self.BANNED_MODULES and node.attr in {"remove", "system", "run", "connect", "socket", "rmtree", "unlink"}:
                self.issues.append(f"Access to '{name}.{node.attr}' is not permitted.")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.BANNED_FUNCTIONS:
            self.issues.append(f"Use of '{node.id}' is not permitted.")
        self.generic_visit(node)

    def _flatten_attribute(self, node: ast.Attribute) -> List[str]:
        parts: List[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.insert(0, current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.insert(0, current.id)
        return parts

    def validate(self, code: str) -> None:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise UnsafeCodeError(f"Syntax error during sandbox analysis: {exc}")
        self.visit(tree)
        if self.issues:
            raise UnsafeCodeError("; ".join(self.issues))


class CodeExecutionSandboxAgent:
    DEFAULT_HISTORY_PATH = "sandbox_execution_history.json"
    MAX_HISTORY = 5000

    def __init__(self, history_path: Optional[str] = None) -> None:
        self.history_path = history_path or os.path.join(os.getcwd(), self.DEFAULT_HISTORY_PATH)
        self.history: List[Dict[str, Any]] = self._load_history()
        self.self_healing = SelfHealingAgent()
        self.experiment_memory = ExperimentMemoryAgent()

    def _load_history(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.history_path):
            return []
        try:
            with open(self.history_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
        except (OSError, json.JSONDecodeError):
            pass
        return []

    def _persist_history(self) -> None:
        directory = os.path.dirname(self.history_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(self.history_path, "w", encoding="utf-8") as fh:
            json.dump(self.history[-self.MAX_HISTORY :], fh, indent=2)

    def execute_code(self, code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = context or {}
        detector = UnsafeCodeDetector()
        try:
            detector.validate(code)
        except UnsafeCodeError as exc:
            output = {
                "success": False,
                "output": "",
                "execution_time": 0.0,
                "warnings": [],
                "error": "Unsafe operation detected",
                "error_type": "UnsafeCode",
                "suggested_fix": str(exc),
                "explanation": "The code contains operations that are not permitted in the sandbox.",
            }
            self._record_history(code, output)
            return output

        exec_globals = self._build_sandbox_globals()
        exec_locals: Dict[str, Any] = {}
        if "dataset" in context and isinstance(context["dataset"], pd.DataFrame):
            exec_locals["df"] = context["dataset"].copy()
            exec_locals["dataset"] = context["dataset"]

        stdout_buffer = io.StringIO()
        warning_list: List[str] = []
        start_time = time.time()
        error_message: Optional[str] = None
        error_type: Optional[str] = None
        suggested_fix: Optional[str] = None
        success = True
        explanation = "Code executed successfully."

        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                old_stdout = os.sys.stdout
                try:
                    os.sys.stdout = stdout_buffer
                    exec(code, exec_globals, exec_locals)
                finally:
                    os.sys.stdout = old_stdout
                warning_list = [str(w.message) for w in caught_warnings]
        except Exception as exc:
            success = False
            error_message = str(exc)
            analysis = self.self_healing.analyze_error(error_message, context.get("dataset_info", {}))
            if isinstance(analysis, dict):
                error_type = analysis.get("error_type", "ExecutionError")
                recovery = self.self_healing.recommend_fix(analysis)
                suggested_fix = recovery.get("recommended_action")
            else:
                error_type = "ExecutionError"
                suggested_fix = None
            explanation = self._build_error_explanation(error_type, suggested_fix)

        end_time = time.time()
        execution_time = round(end_time - start_time, 4)
        output_text = stdout_buffer.getvalue().strip()

        feature_summary = self._detect_new_features(context, exec_locals)
        experiment_summary = None
        if feature_summary:
            experiment_summary = self._record_feature_experiment(feature_summary, context, exec_locals)
        if success and not feature_summary:
            self._record_possible_experiment(context, exec_locals)

        result = {
            "success": success,
            "output": output_text,
            "execution_time": execution_time,
            "warnings": warning_list,
            "error": error_message,
            "error_type": error_type,
            "suggested_fix": suggested_fix,
            "explanation": explanation,
            "experiment_summary": experiment_summary,
        }
        self._record_history(code, result)
        return result

    def _build_sandbox_globals(self) -> Dict[str, Any]:
        import math
        import statistics

        safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bin": bin,
            "bool": bool,
            "bytearray": bytearray,
            "bytes": bytes,
            "chr": chr,
            "complex": complex,
            "dict": dict,
            "divmod": divmod,
            "float": float,
            "format": format,
            "frozenset": frozenset,
            "hash": hash,
            "hex": hex,
            "int": int,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "pow": pow,
            "print": print,
            "range": range,
            "repr": repr,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "type": type,
            "zip": zip,
            "__import__": self._safe_import,
        }
        globals_dict: Dict[str, Any] = {
            "__builtins__": safe_builtins,
            "pd": pd,
            "pandas": pd,
            "np": __import__("numpy"),
            "numpy": __import__("numpy"),
            "math": math,
            "statistics": statistics,
        }
        try:
            import sklearn  # type: ignore
            globals_dict["sklearn"] = sklearn
        except Exception:
            pass
        try:
            import matplotlib  # type: ignore
            globals_dict["matplotlib"] = matplotlib
        except Exception:
            pass
        try:
            import seaborn  # type: ignore
            globals_dict["seaborn"] = seaborn
        except Exception:
            pass
        return globals_dict

    def _safe_import(self, name: str, globals: Any = None, locals: Any = None, fromlist: Tuple[str, ...] = (), level: int = 0) -> Any:
        root_name = name.split(".")[0]
        if root_name not in UnsafeCodeDetector.ALLOWED_MODULES:
            raise ImportError(f"Import of module '{name}' is not permitted in sandbox.")
        return __import__(name, globals, locals, fromlist, level)

    def _record_history(self, code: str, result: Dict[str, Any]) -> None:
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "code": code,
            "success": bool(result.get("success")),
            "execution_time": result.get("execution_time", 0.0),
            "warnings": result.get("warnings", []),
            "error": result.get("error"),
            "error_type": result.get("error_type"),
            "suggested_fix": result.get("suggested_fix"),
        }
        self.history.append(entry)
        self.history = self.history[-self.MAX_HISTORY :]
        self._persist_history()

    def _detect_new_features(self, context: Dict[str, Any], exec_locals: Dict[str, Any]) -> Optional[List[str]]:
        df = exec_locals.get("df")
        if not isinstance(df, pd.DataFrame):
            return None
        original_df = context.get("dataset")
        if not isinstance(original_df, pd.DataFrame):
            return None
        original_cols = set(original_df.columns)
        new_cols = [col for col in df.columns if col not in original_cols]
        return new_cols if new_cols else None

    def _record_feature_experiment(self, new_features: List[str], context: Dict[str, Any], exec_locals: Dict[str, Any]) -> Dict[str, Any]:
        df = exec_locals.get("df")
        data_shape = [df.shape[0], df.shape[1]] if isinstance(df, pd.DataFrame) else []
        notes = f"Created new feature(s): {', '.join(new_features)}."
        summary = self.experiment_memory.log_experiment(
            dataset_name=context.get("dataset_name", "sandbox_dataset"),
            dataset_shape=data_shape,
            algorithm_name="sandbox_feature_engineering",
            feature_count=df.shape[1] if isinstance(df, pd.DataFrame) else None,
            feature_engineering_steps=new_features,
            notes=notes,
        )
        return summary

    def _record_possible_experiment(self, context: Dict[str, Any], exec_locals: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        metrics = {}
        for key in ["accuracy", "score", "test_score", "train_score"]:
            if key in exec_locals and isinstance(exec_locals[key], (int, float)):
                metrics[key] = float(exec_locals[key])
        if metrics:
            df = exec_locals.get("df")
            data_shape = [df.shape[0], df.shape[1]] if isinstance(df, pd.DataFrame) else []
            summary = self.experiment_memory.log_experiment(
                dataset_name=context.get("dataset_name", "sandbox_dataset"),
                dataset_shape=data_shape,
                algorithm_name="sandbox_code_run",
                train_score=metrics.get("train_score"),
                test_score=metrics.get("test_score") or metrics.get("accuracy"),
                notes=f"Recorded metric(s) from sandbox code: {metrics}",
            )
            return summary
        return None

    def _build_error_explanation(self, error_type: Optional[str], suggested_fix: Optional[str]) -> str:
        parts: List[str] = []
        if error_type:
            parts.append(f"The sandbox detected an issue classified as {error_type}.")
        if suggested_fix:
            parts.append(f"Suggested fix: {suggested_fix}")
        if not parts:
            return "The code failed to execute in the sandbox."
        return " ".join(parts)

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
import datetime
from typing import Any, Dict, List, Optional


class SecurityAuthAgent:
    DEFAULT_USERS = "users.json"
    DEFAULT_SESSIONS = "sessions.json"
    DEFAULT_API_KEYS = "api_keys.json"
    DEFAULT_AUDIT = "security_audit_log.json"

    FAILED_LOGIN_LIMIT = 5
    LOCK_DURATION_SECONDS = 15 * 60
    SESSION_DURATION_SECONDS = 24 * 3600
    API_KEY_DURATION_SECONDS = 90 * 24 * 3600
    AUDIT_MAX = 100000

    ROLE_PERMISSIONS = {
        "Admin": {
            "dataset_upload",
            "model_training",
            "deployment",
            "delete_project",
            "manage_users",
            "manage_plugins",
            "view_logs",
        },
        "Data Scientist": {
            "dataset_upload",
            "model_training",
            "experiment_tracking",
            "feature_engineering",
            "deployment_request",
        },
        "Analyst": {"dataset_upload", "view_reports", "run_predictions", "generate_documentation"},
        "Viewer": {"view_reports", "view_models", "view_projects"},
        "API Client": {"api_prediction", "api_status"},
    }

    def __init__(self, users_path: Optional[str] = None, sessions_path: Optional[str] = None,
                 api_keys_path: Optional[str] = None, audit_path: Optional[str] = None):
        self.users_path = users_path or self.DEFAULT_USERS
        self.sessions_path = sessions_path or self.DEFAULT_SESSIONS
        self.api_keys_path = api_keys_path or self.DEFAULT_API_KEYS
        self.audit_path = audit_path or self.DEFAULT_AUDIT

        self._ensure_files()
        self._load_all()

    def _now(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def _ensure_files(self) -> None:
        for path in (self.users_path, self.sessions_path, self.api_keys_path, self.audit_path):
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as fh:
                    if path == self.audit_path:
                        json.dump({"events": []}, fh)
                    else:
                        json.dump({}, fh)

    def _load_all(self) -> None:
        self.users = self._load_json(self.users_path) or {}
        self.sessions = self._load_json(self.sessions_path) or {}
        self.api_keys = self._load_json(self.api_keys_path) or {}
        a = self._load_json(self.audit_path) or {}
        self.audit = a.get("events", []) if isinstance(a, dict) else []

    def _load_json(self, path: str) -> Any:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    def _save_json(self, path: str, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def _hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    # User management
    def create_user(self, username: str, email: str, password: str, role: str = "Viewer") -> Dict[str, Any]:
        if username in (u.get("username") for u in self.users.values()):
            raise ValueError("username already exists")

        user_id = f"USR_{secrets.token_hex(6)}"
        now = self._now()
        pwd_hash = self._hash_password(password)
        user = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "password_hash": pwd_hash,
            "role": role,
            "created_at": now,
            "last_login": None,
            "status": "active",
            "failed_attempts": 0,
            "locked_until": None,
        }
        self.users[user_id] = user
        self._save_json(self.users_path, self.users)
        self.record_security_event("user_created", user_id, f"user {username} created", "INFO")
        return user

    # Authentication
    def authenticate(self, username: str, password: str) -> Dict[str, Any]:
        # find user
        user = next((u for u in self.users.values() if u.get("username") == username), None)
        if not user:
            self.record_security_event("login_failed", None, f"unknown user {username}", "WARNING")
            return {"authenticated": False}

        # check lock
        if user.get("locked_until"):
            try:
                locked_ts = datetime.datetime.fromisoformat(user["locked_until"])
                if locked_ts > datetime.datetime.now(datetime.timezone.utc):
                    self.record_security_event("login_locked", user.get("user_id"), "account locked", "WARNING")
                    return {"authenticated": False, "reason": "locked"}
                else:
                    user["locked_until"] = None
                    user["failed_attempts"] = 0
            except Exception:
                pass

        if self._hash_password(password) == user.get("password_hash"):
            user["failed_attempts"] = 0
            user["last_login"] = self._now()
            self._save_json(self.users_path, self.users)
            session = self.create_session(user.get("user_id"))
            self.record_security_event("login_success", user.get("user_id"), "user authenticated", "INFO")
            return {"authenticated": True, "user_id": user.get("user_id"), "role": user.get("role"), "token": session.get("session_id")}
        else:
            user["failed_attempts"] = user.get("failed_attempts", 0) + 1
            if user["failed_attempts"] >= self.FAILED_LOGIN_LIMIT:
                locked_until = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=self.LOCK_DURATION_SECONDS)).isoformat()
                user["locked_until"] = locked_until
                self.record_security_event("account_locked", user.get("user_id"), "too many failed logins", "HIGH")
            else:
                self.record_security_event("login_failed", user.get("user_id"), "invalid password", "WARNING")
            self._save_json(self.users_path, self.users)
            return {"authenticated": False}

    # Sessions
    def create_session(self, user_id: str) -> Dict[str, Any]:
        session_id = f"S_{secrets.token_hex(16)}"
        now_ts = int(time.time())
        expires = now_ts + self.SESSION_DURATION_SECONDS
        s = {"session_id": session_id, "user_id": user_id, "created_at": now_ts, "expires_at": expires, "active": True}
        self.sessions[session_id] = s
        self._save_json(self.sessions_path, self.sessions)
        return s

    def validate_session(self, session_id: str) -> bool:
        s = self.sessions.get(session_id)
        if not s or not s.get("active"):
            return False
        if int(time.time()) > int(s.get("expires_at", 0)):
            s["active"] = False
            self._save_json(self.sessions_path, self.sessions)
            return False
        return True

    def logout(self, session_id: str) -> bool:
        s = self.sessions.get(session_id)
        if not s:
            return False
        s["active"] = False
        self._save_json(self.sessions_path, self.sessions)
        self.record_security_event("logout", s.get("user_id"), "session ended", "INFO")
        return True

    # RBAC
    def check_permission(self, user_role: str, action: str) -> bool:
        perms = self.ROLE_PERMISSIONS.get(user_role, set())
        return action in perms

    # API keys
    def generate_api_key(self, user_id: str) -> Dict[str, Any]:
        raw_key = secrets.token_hex(32)
        key_hash = self._hash_key(raw_key)
        key_id = f"KEY_{secrets.token_hex(6)}"
        created_at = int(time.time())
        expires_at = created_at + self.API_KEY_DURATION_SECONDS
        entry = {"key_id": key_id, "user_id": user_id, "key_hash": key_hash, "created_at": created_at, "expires_at": expires_at, "active": True}
        self.api_keys[key_id] = entry
        self._save_json(self.api_keys_path, self.api_keys)
        self.record_security_event("api_key_created", user_id, f"key {key_id} created", "INFO")
        return {"key_id": key_id, "key": raw_key}

    def validate_api_key(self, raw_key: str) -> Optional[Dict[str, Any]]:
        h = self._hash_key(raw_key)
        for k, v in list(self.api_keys.items()):
            if v.get("key_hash") == h and v.get("active"):
                if int(time.time()) > int(v.get("expires_at", 0)):
                    v["active"] = False
                    self._save_json(self.api_keys_path, self.api_keys)
                    self.record_security_event("api_key_expired", v.get("user_id"), f"key {k} expired", "WARNING")
                    return None
                self.record_security_event("api_key_validated", v.get("user_id"), f"key {k} used", "INFO")
                return v
        return None

    def revoke_api_key(self, key_id: str) -> bool:
        v = self.api_keys.get(key_id)
        if not v:
            return False
        v["active"] = False
        self._save_json(self.api_keys_path, self.api_keys)
        self.record_security_event("api_key_revoked", v.get("user_id"), f"key {key_id} revoked", "INFO")
        return True

    # Audit log
    def record_security_event(self, event_type: str, user_id: Optional[str], details: str, severity: str = "INFO") -> None:
        entry = {"timestamp": self._now(), "event_type": event_type, "user_id": user_id, "details": details, "severity": severity}
        self.audit.append(entry)
        # trim
        if len(self.audit) > self.AUDIT_MAX:
            self.audit = self.audit[-self.AUDIT_MAX:]
        self._save_json(self.audit_path, {"events": self.audit})

    def generate_security_report(self) -> Dict[str, Any]:
        total_users = len(self.users)
        active_sessions = sum(1 for s in self.sessions.values() if s.get("active"))
        active_api_keys = sum(1 for k in self.api_keys.values() if k.get("active"))
        failed_logins = sum(1 for u in self.users.values() if u.get("failed_attempts", 0) > 0)
        locked_accounts = sum(1 for u in self.users.values() if u.get("locked_until"))
        recent = list(self.audit[-10:])
        # simple security score heuristic
        score = 100 - (locked_accounts * 5) - (failed_logins * 1)
        score = max(0, min(100, score))
        return {
            "total_users": total_users,
            "active_sessions": active_sessions,
            "active_api_keys": active_api_keys,
            "failed_logins": failed_logins,
            "locked_accounts": locked_accounts,
            "security_score": score,
            "recent_events": recent,
        }

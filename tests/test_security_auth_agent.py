import json
import time
from pathlib import Path

from agents.security_auth_agent import SecurityAuthAgent


def test_user_creation_and_duplicate(tmp_path):
    users = tmp_path / "users.json"
    sa = SecurityAuthAgent(users_path=str(users), sessions_path=str(tmp_path / "sessions.json"), api_keys_path=str(tmp_path / "api_keys.json"), audit_path=str(tmp_path / "audit.json"))
    u = sa.create_user("alice", "alice@example.com", "password123", role="Admin")
    assert u["username"] == "alice"
    assert "password_hash" in u
    # duplicate username rejected
    try:
        sa.create_user("alice", "a2@example.com", "x")
        assert False, "duplicate username not rejected"
    except ValueError:
        assert True


def test_authentication_and_lockout(tmp_path):
    sa = SecurityAuthAgent(users_path=str(tmp_path / "u2.json"), sessions_path=str(tmp_path / "s2.json"), api_keys_path=str(tmp_path / "k2.json"), audit_path=str(tmp_path / "a2.json"))
    user = sa.create_user("bob", "bob@example.com", "secret", role="Viewer")
    # wrong password attempts
    for i in range(4):
        r = sa.authenticate("bob", "wrongpass")
        assert not r.get("authenticated")
    # fifth attempt triggers lock
    r = sa.authenticate("bob", "wrongpass")
    assert not r.get("authenticated")
    u = sa.users[user["user_id"]]
    assert u.get("locked_until") is not None
    # correct password while locked should fail
    r2 = sa.authenticate("bob", "secret")
    assert not r2.get("authenticated")


def test_sessions_and_logout(tmp_path):
    sa = SecurityAuthAgent(users_path=str(tmp_path / "u3.json"), sessions_path=str(tmp_path / "s3.json"), api_keys_path=str(tmp_path / "k3.json"), audit_path=str(tmp_path / "a3.json"))
    u = sa.create_user("carl", "carl@example.com", "pw", role="Data Scientist")
    auth = sa.authenticate("carl", "pw")
    assert auth.get("authenticated")
    token = auth.get("token")
    assert sa.validate_session(token)
    sa.logout(token)
    assert not sa.validate_session(token)


def test_rbac_permissions(tmp_path):
    sa = SecurityAuthAgent(users_path=str(tmp_path / "u4.json"), sessions_path=str(tmp_path / "s4.json"), api_keys_path=str(tmp_path / "k4.json"), audit_path=str(tmp_path / "a4.json"))
    assert sa.check_permission("Admin", "manage_users")
    assert not sa.check_permission("Viewer", "deployment")


def test_api_key_generation_and_validation_and_revoke(tmp_path):
    sa = SecurityAuthAgent(users_path=str(tmp_path / "u5.json"), sessions_path=str(tmp_path / "s5.json"), api_keys_path=str(tmp_path / "k5.json"), audit_path=str(tmp_path / "a5.json"))
    u = sa.create_user("dan", "d@example.com", "pw", role="API Client")
    key = sa.generate_api_key(u["user_id"])
    assert "key" in key
    validated = sa.validate_api_key(key["key"])
    assert validated is not None
    assert sa.revoke_api_key(key["key_id"]) is True
    assert sa.validate_api_key(key["key"]) is None


def test_audit_log_and_report_and_limit(tmp_path):
    sa = SecurityAuthAgent(users_path=str(tmp_path / "u6.json"), sessions_path=str(tmp_path / "s6.json"), api_keys_path=str(tmp_path / "k6.json"), audit_path=str(tmp_path / "a6.json"))
    u = sa.create_user("erin", "e@example.com", "pw", role="Viewer")
    sa.record_security_event("test_event", u["user_id"], "details", "INFO")
    report = sa.generate_security_report()
    assert report["total_users"] >= 1
    assert isinstance(report["recent_events"], list)
    # test trimming
    for i in range(105):
        sa.record_security_event("x", None, f"{i}", "INFO")
    # ensure audit is not empty and trimmed to <= AUDIT_MAX
    assert len(sa.audit) <= sa.AUDIT_MAX

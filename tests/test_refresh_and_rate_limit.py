"""
Week 5 tests: refresh-token flow, logout, and login rate-limiting.

Same tmp_path SQLite fixture pattern as the other test files. Note: the
rate limiter's state (app/rate_limit.py) lives in a module-level dict for
the whole test *process*, not per-fixture — so every test here uses its own
unique email to avoid one test's failed attempts bleeding into another's.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app import rate_limit
from app.database import Base, get_db, make_engine
from app.main import app


@pytest.fixture()
def client(tmp_path):
    db_path = tmp_path / "test.db"
    engine = make_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()
    engine.dispose()


def _signup_and_login(client: TestClient, email: str, password: str = "testpass123"):
    client.post("/auth/signup", json={"email": email, "password": password})
    resp = client.post("/auth/login", data={"username": email, "password": password})
    return resp.json()


def test_login_returns_both_access_and_refresh_tokens(client):
    tokens = _signup_and_login(client, "tokens@example.com")
    assert "access_token" in tokens
    assert "refresh_token" in tokens
    assert tokens["access_token"] != tokens["refresh_token"]


def test_refresh_issues_a_new_working_access_token(client):
    tokens = _signup_and_login(client, "refresh@example.com")

    resp = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert resp.status_code == 200
    new_tokens = resp.json()
    assert new_tokens["access_token"] != tokens["access_token"]

    # New access token actually works against a protected endpoint.
    me = client.get(
        "/accounts/me", headers={"Authorization": f"Bearer {new_tokens['access_token']}"}
    )
    assert me.status_code == 200


def test_refresh_token_is_single_use(client):
    """Rotation: using a refresh token revokes it, so reusing it fails."""
    tokens = _signup_and_login(client, "rotate@example.com")

    first = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert first.status_code == 200

    second = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert second.status_code == 401


def test_refresh_with_garbage_token_returns_401(client):
    resp = client.post("/auth/refresh", json={"refresh_token": "not-a-real-token"})
    assert resp.status_code == 401


def test_logout_revokes_refresh_token(client):
    tokens = _signup_and_login(client, "logout@example.com")

    logout_resp = client.post("/auth/logout", json={"refresh_token": tokens["refresh_token"]})
    assert logout_resp.status_code == 204

    refresh_resp = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert refresh_resp.status_code == 401


def test_logout_with_unknown_token_does_not_error(client):
    resp = client.post("/auth/logout", json={"refresh_token": "never-issued"})
    assert resp.status_code == 204


def test_login_rate_limit_blocks_after_repeated_failures(client):
    email = "ratelimited@example.com"
    client.post("/auth/signup", json={"email": email, "password": "correctpass123"})
    rate_limit.clear_attempts(email)  # isolate from any prior test touching this identifier

    for _ in range(rate_limit.MAX_ATTEMPTS):
        resp = client.post("/auth/login", data={"username": email, "password": "wrongpass"})
        assert resp.status_code == 401

    blocked = client.post("/auth/login", data={"username": email, "password": "wrongpass"})
    assert blocked.status_code == 429

    # Even the *correct* password is blocked once the limit is hit — that's
    # the point: it protects the account, not just bad guesses.
    still_blocked = client.post("/auth/login", data={"username": email, "password": "correctpass123"})
    assert still_blocked.status_code == 429

    rate_limit.clear_attempts(email)  # don't leak state into later tests


def test_successful_login_clears_previous_failed_attempts(client):
    email = "recovers@example.com"
    client.post("/auth/signup", json={"email": email, "password": "correctpass123"})
    rate_limit.clear_attempts(email)

    for _ in range(rate_limit.MAX_ATTEMPTS - 1):
        client.post("/auth/login", data={"username": email, "password": "wrongpass"})

    good = client.post("/auth/login", data={"username": email, "password": "correctpass123"})
    assert good.status_code == 200

    # A subsequent login should not be blocked — the successful login above
    # should have cleared the counter, not just added to it.
    good_again = client.post("/auth/login", data={"username": email, "password": "correctpass123"})
    assert good_again.status_code == 200
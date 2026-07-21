"""
Week 3 tests: concurrency safety and idempotency.

Each test builds its own temp SQLite file and wires it in via FastAPI's
dependency_overrides, so these tests don't touch wallet.db or interfere with
tests in test_wallet_flow.py. A real file (not ":memory:") is used
deliberately: SQLite in-memory databases are private per connection unless
you set up shared cache, and these tests need multiple threads to share one
database the way multiple real clients would.
"""

import threading

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

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

    # Release SQLAlchemy's pooled connections so the file handle is freed.
    # We deliberately do NOT delete db_path ourselves — it lives under
    # pytest's tmp_path, which pytest cleans up on its own schedule and
    # never fails a test over a lingering OS file lock (unlike a manual
    # os.remove would on Windows).
    engine.dispose()


def _signup_and_login(client: TestClient, email: str, password: str = "testpass123"):
    client.post("/auth/signup", json={"email": email, "password": password})
    resp = client.post("/auth/login", data={"username": email, "password": password})
    token = resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    account_id = client.get("/accounts/me", headers=headers).json()[0]["id"]
    return headers, account_id


def test_concurrent_withdrawals_only_one_succeeds(client):
    """
    Fire 5 simultaneous withdrawals of the full balance. Without row
    locking, several could read "balance = 100" before any of them commits,
    and all pass the insufficient-balance check. With locking, exactly one
    should succeed and the rest should see the post-withdrawal balance.
    """
    headers, account_id = _signup_and_login(client, "race@example.com")
    client.post(f"/accounts/{account_id}/deposit", json={"amount": 100}, headers=headers)

    results = []

    def do_withdraw():
        resp = client.post(f"/accounts/{account_id}/withdraw", json={"amount": 100}, headers=headers)
        results.append(resp.status_code)

    threads = [threading.Thread(target=do_withdraw) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(200) == 1
    assert results.count(400) == 4

    balance = client.get("/accounts/me", headers=headers).json()[0]["balance"]
    assert float(balance) == 0.0


def test_idempotent_deposit_returns_cached_response(client):
    """Same key, sent twice sequentially -> same transaction, charged once."""
    headers, account_id = _signup_and_login(client, "idem@example.com")
    key = "deposit-key-123"

    resp1 = client.post(
        f"/accounts/{account_id}/deposit",
        json={"amount": 50},
        headers={**headers, "Idempotency-Key": key},
    )
    resp2 = client.post(
        f"/accounts/{account_id}/deposit",
        json={"amount": 50},
        headers={**headers, "Idempotency-Key": key},
    )

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json()["id"] == resp2.json()["id"]

    balance = client.get("/accounts/me", headers=headers).json()[0]["balance"]
    assert float(balance) == 50.0


def test_deposit_without_idempotency_key_is_not_deduplicated(client):
    """No key supplied -> old behavior: each request is its own deposit."""
    headers, account_id = _signup_and_login(client, "no-key@example.com")

    client.post(f"/accounts/{account_id}/deposit", json={"amount": 50}, headers=headers)
    client.post(f"/accounts/{account_id}/deposit", json={"amount": 50}, headers=headers)

    balance = client.get("/accounts/me", headers=headers).json()[0]["balance"]
    assert float(balance) == 100.0


def test_concurrent_duplicate_deposits_same_key_charged_once(client):
    """
    The race this actually protects against: a client's retry logic fires a
    second request (same key) before the first one has finished. Both hit
    the server at nearly the same instant.
    """
    headers, account_id = _signup_and_login(client, "idem-race@example.com")
    key = "concurrent-deposit-key"
    results = []

    def do_deposit():
        resp = client.post(
            f"/accounts/{account_id}/deposit",
            json={"amount": 20},
            headers={**headers, "Idempotency-Key": key},
        )
        results.append(resp.status_code)

    threads = [threading.Thread(target=do_deposit) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(code == 200 for code in results)
    balance = client.get("/accounts/me", headers=headers).json()[0]["balance"]
    assert float(balance) == 20.0  # not 100 — only the first request actually charged


def test_idempotent_transfer_returns_cached_response(client):
    from_headers, from_account_id = _signup_and_login(client, "transfer-from@example.com")
    _, to_account_id = _signup_and_login(client, "transfer-to@example.com")
    client.post(f"/accounts/{from_account_id}/deposit", json={"amount": 100}, headers=from_headers)

    key = "transfer-key-abc"
    body = {"to_account_id": to_account_id, "amount": 30}

    resp1 = client.post(
        f"/accounts/{from_account_id}/transfer",
        json=body,
        headers={**from_headers, "Idempotency-Key": key},
    )
    resp2 = client.post(
        f"/accounts/{from_account_id}/transfer",
        json=body,
        headers={**from_headers, "Idempotency-Key": key},
    )

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json()["debit"]["id"] == resp2.json()["debit"]["id"]

    from_balance = client.get("/accounts/me", headers=from_headers).json()[0]["balance"]
    assert float(from_balance) == 70.0  # 100 - 30, only once
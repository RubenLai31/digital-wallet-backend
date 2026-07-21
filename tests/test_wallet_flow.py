"""
Tests for the wallet API.

Uses an in-memory SQLite database so tests never touch your real wallet.db,
and tears it down after each test so tests don't leak state into each other.
"""

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app

# In-memory SQLite, shared across connections via StaticPool. This means no
# test db file is ever created on disk, so there's nothing to clean up and
# no risk of tests leaking state through a real file.
TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture()
def client():
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


def _signup_and_login(client: TestClient, email: str = "you@example.com", password: str = "yourpassword123"):
    """Helper: signup, log in, and return (headers, account_id)."""
    signup_resp = client.post("/auth/signup", json={"email": email, "password": password})
    assert signup_resp.status_code == 201

    login_resp = client.post("/auth/login", data={"username": email, "password": password})
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    accounts = client.get("/accounts/me", headers=headers).json()
    account_id = accounts[0]["id"]

    return headers, account_id


def test_full_signup_login_deposit_withdraw_flow(client: TestClient):
    headers, account_id = _signup_and_login(client)

    deposit_resp = client.post(f"/accounts/{account_id}/deposit", json={"amount": 100}, headers=headers)
    assert deposit_resp.status_code == 200
    assert Decimal(deposit_resp.json()["amount"]) == Decimal("100")

    withdraw_resp = client.post(f"/accounts/{account_id}/withdraw", json={"amount": 40}, headers=headers)
    assert withdraw_resp.status_code == 200
    assert Decimal(withdraw_resp.json()["amount"]) == Decimal("-40")

    balance_resp = client.get("/accounts/me", headers=headers)
    assert Decimal(balance_resp.json()[0]["balance"]) == Decimal("60")


def test_withdraw_more_than_balance_fails(client: TestClient):
    headers, account_id = _signup_and_login(client)

    # Deposit less than we're about to try to withdraw.
    client.post(f"/accounts/{account_id}/deposit", json={"amount": 50}, headers=headers)

    resp = client.post(f"/accounts/{account_id}/withdraw", json={"amount": 100}, headers=headers)

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Insufficient balance"

    # And the balance should be unaffected by the failed withdrawal.
    balance_resp = client.get("/accounts/me", headers=headers)
    assert Decimal(balance_resp.json()[0]["balance"]) == Decimal("50")


def test_concurrent_withdrawals_cannot_overdraft():
    """
    Regression test for the balance-check race condition: without row
    locking, concurrent requests can each read the balance before any of
    them commits, and all pass an "is there enough money" check that should
    have only let one through.

    This needs its own fixture rather than the shared `client` fixture:
    the shared fixture's in-memory SQLite DB uses a single connection
    (StaticPool) shared across all threads, which means there's no real
    per-thread connection for FOR UPDATE to lock — it can't exercise the
    fix at all. A file-based DB gives each thread's session a real
    connection from an actual connection pool, so SQLite's locking (and by
    extension, Postgres's row locking in production) is actually exercised.
    """
    import os
    import threading

    from sqlalchemy.orm import sessionmaker

    from app.database import make_engine

    db_path = "./_test_concurrency.db"
    engine = make_engine(f"sqlite:///{db_path}")
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            headers, account_id = _signup_and_login(client)
            client.post(f"/accounts/{account_id}/deposit", json={"amount": 100}, headers=headers)

            results = []
            lock = threading.Lock()

            def withdraw():
                resp = client.post(
                    f"/accounts/{account_id}/withdraw", json={"amount": 60}, headers=headers
                )
                with lock:
                    results.append(resp.status_code)

            threads = [threading.Thread(target=withdraw) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Started with 100; three concurrent attempts to withdraw 60 each.
            # Exactly one can succeed (leaving 40) — two must fail.
            assert sorted(results) == [200, 400, 400]

            balance = Decimal(client.get("/accounts/me", headers=headers).json()[0]["balance"])
            assert balance == Decimal("40")
            assert balance >= 0
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()
        if os.path.exists(db_path):
            os.remove(db_path)


def test_signup_with_password_over_72_bytes_fails(client: TestClient):
    """
    bcrypt silently truncates passwords past 72 bytes, which would make two
    different passwords sharing the same first 72 bytes hash identically.
    The API should reject long passwords outright instead of allowing that.
    """
    resp = client.post(
        "/auth/signup",
        json={"email": "longpw@example.com", "password": "a" * 73},
    )
    assert resp.status_code == 422


def test_signup_with_duplicate_email_fails(client: TestClient):
    email = "duplicate@example.com"
    first = client.post("/auth/signup", json={"email": email, "password": "yourpassword123"})
    assert first.status_code == 201

    second = client.post("/auth/signup", json={"email": email, "password": "differentpassword456"})

    assert second.status_code == 400
    assert second.json()["detail"] == "Email already registered"
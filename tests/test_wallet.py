"""
Tests for the wallet API.

Uses an in-memory SQLite database for most tests (fast, fully isolated),
and a real file-based database with its own connection pool for tests that
exercise genuine concurrency — see the module docstrings on those tests for
why: in-memory SQLite with StaticPool shares ONE connection across every
thread, which can't exercise real row locking at all.
"""

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app import models

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
    """Helper: signup, log in, and return (headers, account_id, tokens)."""
    signup_resp = client.post("/auth/signup", json={"email": email, "password": password})
    assert signup_resp.status_code == 201

    login_resp = client.post("/auth/login", data={"username": email, "password": password})
    assert login_resp.status_code == 200
    tokens = login_resp.json()
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    accounts = client.get("/accounts/me", headers=headers).json()
    account_id = accounts[0]["id"]

    return headers, account_id, tokens


def _promote_to_admin(email: str) -> None:
    """Test-only equivalent of scripts/make_admin.py — no public API for this, by design."""
    db = TestingSessionLocal()
    try:
        user = db.query(models.User).filter(models.User.email == email).first()
        user.is_admin = True
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Core flows
# ---------------------------------------------------------------------------


def test_full_signup_login_deposit_withdraw_flow(client: TestClient):
    headers, account_id, tokens = _signup_and_login(client)
    assert "refresh_token" in tokens

    deposit_resp = client.post(f"/accounts/{account_id}/deposit", json={"amount": 100}, headers=headers)
    assert deposit_resp.status_code == 200
    assert Decimal(deposit_resp.json()["amount"]) == Decimal("100")
    assert deposit_resp.json()["status"] == "approved"

    withdraw_resp = client.post(f"/accounts/{account_id}/withdraw", json={"amount": 40}, headers=headers)
    assert withdraw_resp.status_code == 200
    assert Decimal(withdraw_resp.json()["amount"]) == Decimal("-40")

    balance_resp = client.get("/accounts/me", headers=headers)
    assert Decimal(balance_resp.json()[0]["balance"]) == Decimal("60")


def test_withdraw_more_than_balance_fails(client: TestClient):
    headers, account_id, _ = _signup_and_login(client)
    client.post(f"/accounts/{account_id}/deposit", json={"amount": 50}, headers=headers)

    resp = client.post(f"/accounts/{account_id}/withdraw", json={"amount": 100}, headers=headers)
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Insufficient balance"

    balance_resp = client.get("/accounts/me", headers=headers)
    assert Decimal(balance_resp.json()[0]["balance"]) == Decimal("50")


def test_signup_with_password_over_72_bytes_fails(client: TestClient):
    resp = client.post("/auth/signup", json={"email": "longpw@example.com", "password": "a" * 73})
    assert resp.status_code == 422


def test_signup_with_duplicate_email_fails(client: TestClient):
    email = "duplicate@example.com"
    first = client.post("/auth/signup", json={"email": email, "password": "yourpassword123"})
    assert first.status_code == 201

    second = client.post("/auth/signup", json={"email": email, "password": "differentpassword456"})
    assert second.status_code == 400
    assert second.json()["detail"] == "Email already registered"


# ---------------------------------------------------------------------------
# Refresh tokens / logout
# ---------------------------------------------------------------------------


def test_refresh_token_rotates_and_old_one_stops_working(client: TestClient):
    _, _, tokens = _signup_and_login(client, "refresh@example.com")

    refresh_resp = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert refresh_resp.status_code == 200
    new_tokens = refresh_resp.json()
    assert new_tokens["refresh_token"] != tokens["refresh_token"]

    # The old refresh token was single-use — reusing it should now fail.
    reuse_resp = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert reuse_resp.status_code == 401

    # The new one should still work.
    second_refresh = client.post("/auth/refresh", json={"refresh_token": new_tokens["refresh_token"]})
    assert second_refresh.status_code == 200


def test_logout_revokes_refresh_token(client: TestClient):
    _, _, tokens = _signup_and_login(client, "logout@example.com")

    logout_resp = client.post("/auth/logout", json={"refresh_token": tokens["refresh_token"]})
    assert logout_resp.status_code == 204

    refresh_resp = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert refresh_resp.status_code == 401


def test_login_rate_limited_after_repeated_failures(client: TestClient):
    email = "ratelimited@example.com"
    client.post("/auth/signup", json={"email": email, "password": "correctpassword1"})

    for _ in range(5):
        resp = client.post("/auth/login", data={"username": email, "password": "wrongpassword"})
        assert resp.status_code == 401

    # 6th attempt (even with the CORRECT password) should be blocked by the limiter.
    blocked_resp = client.post("/auth/login", data={"username": email, "password": "correctpassword1"})
    assert blocked_resp.status_code == 429


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_repeated_deposit_with_same_idempotency_key_only_applies_once(client: TestClient):
    headers, account_id, _ = _signup_and_login(client, "idempotent@example.com")
    idem_headers = {**headers, "Idempotency-Key": "deposit-abc-123"}

    first = client.post(f"/accounts/{account_id}/deposit", json={"amount": 50}, headers=idem_headers)
    second = client.post(f"/accounts/{account_id}/deposit", json={"amount": 50}, headers=idem_headers)

    assert first.status_code == second.status_code == 200
    assert first.json()["id"] == second.json()["id"]  # same cached transaction, not two

    balance = Decimal(client.get("/accounts/me", headers=headers).json()[0]["balance"])
    assert balance == Decimal("50")  # not 100 — the retry didn't double-apply


# ---------------------------------------------------------------------------
# Transfers
# ---------------------------------------------------------------------------


def test_transfer_moves_money_between_accounts(client: TestClient):
    a_headers, a_account, _ = _signup_and_login(client, "alice@example.com")
    b_headers, b_account, _ = _signup_and_login(client, "bob@example.com")

    client.post(f"/accounts/{a_account}/deposit", json={"amount": 100}, headers=a_headers)

    resp = client.post(
        f"/accounts/{a_account}/transfer",
        json={"to_account_id": b_account, "amount": 30},
        headers=a_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert Decimal(body["debit"]["amount"]) == Decimal("-30")
    assert Decimal(body["credit"]["amount"]) == Decimal("30")

    a_balance = Decimal(client.get("/accounts/me", headers=a_headers).json()[0]["balance"])
    b_balance = Decimal(client.get("/accounts/me", headers=b_headers).json()[0]["balance"])
    assert a_balance == Decimal("70")
    assert b_balance == Decimal("30")


def test_transfer_with_insufficient_balance_fails(client: TestClient):
    a_headers, a_account, _ = _signup_and_login(client, "alice2@example.com")
    _, b_account, _ = _signup_and_login(client, "bob2@example.com")
    client.post(f"/accounts/{a_account}/deposit", json={"amount": 10}, headers=a_headers)

    resp = client.post(
        f"/accounts/{a_account}/transfer",
        json={"to_account_id": b_account, "amount": 50},
        headers=a_headers,
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Insufficient balance"


def test_transfer_to_same_account_rejected(client: TestClient):
    headers, account_id, _ = _signup_and_login(client, "alice3@example.com")
    client.post(f"/accounts/{account_id}/deposit", json={"amount": 50}, headers=headers)

    resp = client.post(
        f"/accounts/{account_id}/transfer",
        json={"to_account_id": account_id, "amount": 10},
        headers=headers,
    )
    assert resp.status_code == 400


def test_transfer_to_nonexistent_account_fails(client: TestClient):
    headers, account_id, _ = _signup_and_login(client, "alice4@example.com")
    client.post(f"/accounts/{account_id}/deposit", json={"amount": 50}, headers=headers)

    resp = client.post(
        f"/accounts/{account_id}/transfer",
        json={"to_account_id": 999999, "amount": 10},
        headers=headers,
    )
    assert resp.status_code == 404


def test_transfer_from_someone_elses_account_forbidden(client: TestClient):
    a_headers, a_account, _ = _signup_and_login(client, "alice5@example.com")
    _, b_account, _ = _signup_and_login(client, "bob5@example.com")

    resp = client.post(
        f"/accounts/{b_account}/transfer",
        json={"to_account_id": a_account, "amount": 10},
        headers=a_headers,
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Fraud / risk scoring
# ---------------------------------------------------------------------------


def test_small_deposit_stays_approved_and_counts_immediately(client: TestClient):
    """Baseline: ordinary activity shouldn't get flagged (no false positives)."""
    headers, account_id, _ = _signup_and_login(client, "normal@example.com")

    resp = client.post(f"/accounts/{account_id}/deposit", json={"amount": 100}, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "approved"
    assert resp.json()["risk_score"] == 0.0

    balance = Decimal(client.get("/accounts/me", headers=headers).json()[0]["balance"])
    assert balance == Decimal("100")


def test_new_account_moderate_deposit_gets_flagged_and_excluded_from_balance(client: TestClient):
    """
    600 on a brand-new account crosses only the new-account heuristic
    (score 0.4) — enough to flag, not enough to block. The deposit should
    be recorded but NOT counted toward the balance until reviewed.
    """
    headers, account_id, _ = _signup_and_login(client, "newacct@example.com")

    resp = client.post(f"/accounts/{account_id}/deposit", json={"amount": 600}, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "flagged_for_review"
    assert resp.json()["risk_score"] == pytest.approx(0.4)

    balance = Decimal(client.get("/accounts/me", headers=headers).json()[0]["balance"])
    assert balance == Decimal("0")  # held out until reviewed


def test_large_deposit_on_new_account_gets_blocked(client: TestClient):
    """15000 on a brand-new account stacks large-amount (0.5) + new-account
    (0.4) = 0.9, over the block threshold — rejected outright, not just held."""
    headers, account_id, _ = _signup_and_login(client, "bigdeposit@example.com")

    resp = client.post(f"/accounts/{account_id}/deposit", json={"amount": 15000}, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "blocked"

    balance = Decimal(client.get("/accounts/me", headers=headers).json()[0]["balance"])
    assert balance == Decimal("0")


def test_high_velocity_gets_flagged(client: TestClient):
    """5 small (sub-threshold) deposits are all fine individually; the 6th
    in the same window trips the velocity heuristic on its own."""
    headers, account_id, _ = _signup_and_login(client, "velocity@example.com")

    for _ in range(5):
        resp = client.post(f"/accounts/{account_id}/deposit", json={"amount": 10}, headers=headers)
        assert resp.json()["status"] == "approved"

    sixth = client.post(f"/accounts/{account_id}/deposit", json={"amount": 10}, headers=headers)
    assert sixth.json()["status"] == "flagged_for_review"


def test_non_admin_cannot_access_review_endpoints(client: TestClient):
    headers, account_id, _ = _signup_and_login(client, "regularuser@example.com")

    list_resp = client.get("/admin/transactions/flagged", headers=headers)
    assert list_resp.status_code == 403

    review_resp = client.post(
        "/transactions/1/review", json={"decision": "approve"}, headers=headers
    )
    assert review_resp.status_code == 403


def test_admin_can_approve_flagged_deposit(client: TestClient):
    headers, account_id, _ = _signup_and_login(client, "toflag@example.com")
    deposit_resp = client.post(f"/accounts/{account_id}/deposit", json={"amount": 600}, headers=headers)
    txn_id = deposit_resp.json()["id"]
    assert deposit_resp.json()["status"] == "flagged_for_review"

    admin_headers, _, _ = _signup_and_login(client, "admin1@example.com")
    _promote_to_admin("admin1@example.com")
    # Re-login so the fresh token reflects... not actually needed, is_admin
    # is checked fresh from the DB on every request via get_current_user.

    flagged_list = client.get("/admin/transactions/flagged", headers=admin_headers)
    assert flagged_list.status_code == 200
    assert any(t["id"] == txn_id for t in flagged_list.json())

    review_resp = client.post(
        f"/transactions/{txn_id}/review", json={"decision": "approve"}, headers=admin_headers
    )
    assert review_resp.status_code == 200
    assert review_resp.json()["status"] == "approved"

    balance = Decimal(client.get("/accounts/me", headers=headers).json()[0]["balance"])
    assert balance == Decimal("600")  # now counts


def test_admin_can_reject_flagged_deposit(client: TestClient):
    headers, account_id, _ = _signup_and_login(client, "torejectuser@example.com")
    deposit_resp = client.post(f"/accounts/{account_id}/deposit", json={"amount": 600}, headers=headers)
    txn_id = deposit_resp.json()["id"]

    admin_headers, _, _ = _signup_and_login(client, "admin2@example.com")
    _promote_to_admin("admin2@example.com")

    review_resp = client.post(
        f"/transactions/{txn_id}/review", json={"decision": "reject"}, headers=admin_headers
    )
    assert review_resp.status_code == 200
    assert review_resp.json()["status"] == "blocked"

    balance = Decimal(client.get("/accounts/me", headers=headers).json()[0]["balance"])
    assert balance == Decimal("0")  # stays excluded


def test_review_of_flagged_transfer_resolves_both_legs(client: TestClient):
    """
    A flagged transfer produces two rows (debit + credit). Reviewing just
    one of them must not leave the other stuck — this is the atomicity
    property from Week 2 applied to the review step.
    """
    a_headers, a_account, _ = _signup_and_login(client, "transferflag_a@example.com")
    b_headers, b_account, _ = _signup_and_login(client, "transferflag_b@example.com")

    # Get account A's balance above 600 without itself being flagged: two
    # separate small-enough deposits, spaced under every per-deposit
    # threshold, land approved.
    client.post(f"/accounts/{a_account}/deposit", json={"amount": 400}, headers=a_headers)
    client.post(f"/accounts/{a_account}/deposit", json={"amount": 400}, headers=a_headers)
    assert Decimal(client.get("/accounts/me", headers=a_headers).json()[0]["balance"]) == Decimal("800")

    transfer_resp = client.post(
        f"/accounts/{a_account}/transfer",
        json={"to_account_id": b_account, "amount": 600},
        headers=a_headers,
    )
    body = transfer_resp.json()
    assert body["debit"]["status"] == "flagged_for_review"
    assert body["credit"]["status"] == "flagged_for_review"
    debit_id = body["debit"]["id"]
    credit_id = body["credit"]["id"]

    # Neither side should count yet.
    assert Decimal(client.get("/accounts/me", headers=a_headers).json()[0]["balance"]) == Decimal("800")
    assert Decimal(client.get("/accounts/me", headers=b_headers).json()[0]["balance"]) == Decimal("0")

    admin_headers, _, _ = _signup_and_login(client, "admin3@example.com")
    _promote_to_admin("admin3@example.com")

    # Review only the debit leg — the credit leg should resolve too.
    review_resp = client.post(
        f"/transactions/{debit_id}/review", json={"decision": "approve"}, headers=admin_headers
    )
    assert review_resp.status_code == 200

    credit_check = client.get(f"/transactions/{credit_id}", headers=b_headers)
    assert credit_check.json()["status"] == "approved"

    assert Decimal(client.get("/accounts/me", headers=a_headers).json()[0]["balance"]) == Decimal("200")
    assert Decimal(client.get("/accounts/me", headers=b_headers).json()[0]["balance"]) == Decimal("600")


def test_reviewing_already_resolved_transaction_fails(client: TestClient):
    headers, account_id, _ = _signup_and_login(client, "alreadyresolved@example.com")
    deposit_resp = client.post(f"/accounts/{account_id}/deposit", json={"amount": 100}, headers=headers)
    txn_id = deposit_resp.json()["id"]  # already approved, never flagged

    admin_headers, _, _ = _signup_and_login(client, "admin4@example.com")
    _promote_to_admin("admin4@example.com")

    resp = client.post(f"/transactions/{txn_id}/review", json={"decision": "approve"}, headers=admin_headers)
    assert resp.status_code == 400


def test_reviewing_nonexistent_transaction_404s(client: TestClient):
    admin_headers, _, _ = _signup_and_login(client, "admin5@example.com")
    _promote_to_admin("admin5@example.com")

    resp = client.post("/transactions/999999/review", json={"decision": "approve"}, headers=admin_headers)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Concurrency — needs a real file-based DB (see module docstring)
# ---------------------------------------------------------------------------


def test_concurrent_withdrawals_cannot_overdraft():
    import os
    import threading

    from sqlalchemy.orm import sessionmaker as sm

    from app.database import make_engine

    db_path = "./_test_concurrency.db"
    file_engine = make_engine(f"sqlite:///{db_path}")
    FileSessionLocal = sm(autocommit=False, autoflush=False, bind=file_engine)
    Base.metadata.create_all(bind=file_engine)

    def override_get_db():
        db = FileSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as file_client:
            headers, account_id, _ = _signup_and_login(file_client, "concurrency@example.com")
            file_client.post(f"/accounts/{account_id}/deposit", json={"amount": 100}, headers=headers)

            results = []
            lock = threading.Lock()

            def withdraw():
                resp = file_client.post(
                    f"/accounts/{account_id}/withdraw", json={"amount": 60}, headers=headers
                )
                with lock:
                    results.append(resp.status_code)

            threads = [threading.Thread(target=withdraw) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert sorted(results) == [200, 400, 400]
            balance = Decimal(file_client.get("/accounts/me", headers=headers).json()[0]["balance"])
            assert balance == Decimal("40")
            assert balance >= 0
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=file_engine)
        file_engine.dispose()
        if os.path.exists(db_path):
            os.remove(db_path)


def test_opposite_direction_transfers_do_not_deadlock():
    import os
    import threading

    from sqlalchemy.orm import sessionmaker as sm

    from app.database import make_engine

    db_path = "./_test_deadlock.db"
    file_engine = make_engine(f"sqlite:///{db_path}")
    FileSessionLocal = sm(autocommit=False, autoflush=False, bind=file_engine)
    Base.metadata.create_all(bind=file_engine)

    def override_get_db():
        db = FileSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as file_client:
            a_headers, a_account, _ = _signup_and_login(file_client, "deadlock_a@example.com")
            b_headers, b_account, _ = _signup_and_login(file_client, "deadlock_b@example.com")
            file_client.post(f"/accounts/{a_account}/deposit", json={"amount": 100}, headers=a_headers)
            file_client.post(f"/accounts/{b_account}/deposit", json={"amount": 100}, headers=b_headers)

            results = []
            lock = threading.Lock()

            def transfer_a_to_b():
                r = file_client.post(
                    f"/accounts/{a_account}/transfer",
                    json={"to_account_id": b_account, "amount": 10},
                    headers=a_headers,
                )
                with lock:
                    results.append(("a_to_b", r.status_code))

            def transfer_b_to_a():
                r = file_client.post(
                    f"/accounts/{b_account}/transfer",
                    json={"to_account_id": a_account, "amount": 10},
                    headers=b_headers,
                )
                with lock:
                    results.append(("b_to_a", r.status_code))

            threads = [
                threading.Thread(target=transfer_a_to_b),
                threading.Thread(target=transfer_b_to_a),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)
            assert all(not t.is_alive() for t in threads), "transfers appear deadlocked"
            assert sorted(results) == [("a_to_b", 200), ("b_to_a", 200)]
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=file_engine)
        file_engine.dispose()
        if os.path.exists(db_path):
            os.remove(db_path)
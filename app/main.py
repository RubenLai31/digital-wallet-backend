"""
Main FastAPI application.

Endpoints included:
  POST /auth/signup        - create a user + their first account
  POST /auth/login         - get an access token + refresh token
  POST /auth/refresh       - exchange a refresh token for a new access token
  POST /auth/logout        - revoke a refresh token
  GET  /accounts/me        - list my accounts + balances
  POST /accounts/{id}/deposit
  POST /accounts/{id}/withdraw
  POST /accounts/{id}/transfer   - move money to another account, atomically
  GET  /accounts/{id}/transactions
  GET  /transactions/{id}
  GET  /admin/transactions/flagged   - list transactions pending fraud review (admin only)
  POST /transactions/{id}/review     - approve or reject a flagged transaction (admin only)

Run with:
  uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs for interactive API docs (free, from FastAPI).
"""

import json
import logging
import os

from fastapi import FastAPI, Depends, Header, HTTPException, Request, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm

from . import models, schemas, auth, rate_limit, risk
from .database import engine, get_db

# Production logging: captures unhandled exceptions so they're visible in
# Render's logs (or wherever stdout/stderr is collected). Without this,
# a 500 is returned to the client but the actual error is silent.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Digital Wallet Backend", version="0.1.0")

# CORS: the frontend's origin(s) must be listed here or browsers will block
# every cross-origin request. In production, set CORS_ORIGINS to your actual
# frontend domain(s) (comma-separated). The default covers local dev only —
# Vite/React typically runs on :3000 or :5173. Never default to ["*"] in
# production; that lets any website make authenticated requests to your API.
_cors_origins_raw = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Liveness probe for Render (or any PaaS) health checks."""
    return {"status": "ok"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Catch-all for unexpected errors: logs the full stack trace server-side
    (so production issues are debuggable from logs), but returns a clean
    generic 500 to the client — never leak stack traces, SQL errors, or
    internal paths to API consumers.
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.post("/auth/signup", response_model=schemas.UserOut, status_code=status.HTTP_201_CREATED)
def signup(user_in: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.email == user_in.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = models.User(email=user_in.email, hashed_password=auth.hash_password(user_in.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        # The check above is just a fast path — it doesn't stop two concurrent
        # signups with the same email from both passing it. The database's
        # unique constraint on `email` is what actually guarantees no
        # duplicates; this turns that guarantee into a clean 400 instead of
        # an unhandled 500.
        db.rollback()
        raise HTTPException(status_code=400, detail="Email already registered")
    db.refresh(user)

    # Every new user gets one default account to start with.
    account = models.Account(owner_id=user.id, currency="USD")
    db.add(account)
    db.commit()

    return user


@app.post("/auth/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Scoped by email rather than IP: an IP-based limit is easy for an
    # attacker to route around with multiple source addresses, but the
    # thing actually worth protecting — one specific account — is what
    # this stops regardless of how many IPs the attempts come from.
    rate_limit.check_rate_limit(form_data.username)

    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        rate_limit.record_failed_attempt(form_data.username)
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    rate_limit.clear_attempts(form_data.username)
    access_token = auth.create_access_token(data={"sub": str(user.id)})
    refresh_token = auth.issue_refresh_token(db, user.id)
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}


@app.post("/auth/refresh", response_model=schemas.Token)
def refresh(body: schemas.RefreshRequest, db: Session = Depends(get_db)):
    record = auth.get_valid_refresh_token(db, body.refresh_token)
    if record is None:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    # Rotate: revoke the one just used and issue a brand new one, rather
    # than reusing it. This means a refresh token is single-use — if a
    # leaked token gets used by both an attacker and the real client, the
    # second use fails outright instead of silently succeeding, which is a
    # detectable signal that something's wrong (the standard argument for
    # rotation over long-lived reusable refresh tokens).
    record.revoked = True
    db.commit()

    new_access_token = auth.create_access_token(data={"sub": str(record.user_id)})
    new_refresh_token = auth.issue_refresh_token(db, record.user_id)
    return {"access_token": new_access_token, "refresh_token": new_refresh_token, "token_type": "bearer"}


@app.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(body: schemas.RefreshRequest, db: Session = Depends(get_db)):
    # Only the refresh token is revoked. The access token, if one is still
    # live, keeps working until it naturally expires (up to 15 minutes) —
    # that's the tradeoff of using stateless JWTs for access tokens (see
    # the comment on ACCESS_TOKEN_EXPIRE_MINUTES in auth.py). What logout
    # *does* guarantee is that no new access token can be minted from this
    # session afterward.
    auth.revoke_refresh_token(db, body.refresh_token)
    return None


@app.get("/accounts/me", response_model=list[schemas.AccountOut])
def list_my_accounts(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    return db.query(models.Account).filter(models.Account.owner_id == current_user.id).all()


def _get_owned_account(
    account_id: int, current_user: models.User, db: Session, for_update: bool = False
) -> models.Account:
    """
    for_update=True locks the account row (SELECT ... FOR UPDATE) for the rest
    of the current transaction. Use this for anything that reads the balance
    and then decides whether to write a new transaction (deposit, withdraw,
    transfer) — without it, two concurrent requests can both read the balance
    before either commits, and both pass an "is there enough money" check
    that should have only let one of them through.

    Locking the account row works as a stand-in for locking "this account's
    transaction history", even though the balance itself is never stored:
    a second request's SELECT ... FOR UPDATE on the same account blocks until
    the first request commits (or rolls back), so by the time it proceeds,
    the transactions relationship reflects the first request's write.

    SQLite ignores FOR UPDATE (it has no row-level locking — it locks the
    whole database file instead, which is coarser but happens to prevent the
    same race locally). This matters once you're on Postgres, which is real
    row-level locking and is where this actually earns its keep.

    Only pass for_update=True for endpoints that mutate balance-affecting
    state. Read-only endpoints (like GET /accounts/me) should leave it False
    to avoid taking locks they don't need.
    """
    query = db.query(models.Account).filter(models.Account.id == account_id)
    if for_update:
        query = query.with_for_update()
    account = query.first()
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    if account.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your account")
    return account


def get_idempotency_key(
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> str | None:
    """
    Optional header, opt-in per request. Clients that want retry-safety pass
    a unique key (e.g. a UUID they generate once per logical operation and
    reuse on every retry of that same operation); clients that don't pass
    one get the old unprotected behavior.
    """
    return idempotency_key


def _get_cached_idempotent_response(
    db: Session, user_id: int, key: str | None, endpoint: str
) -> dict | None:
    """
    Look up a previously-cached response for this (user, key, endpoint).
    Returns None if there's no key to check, or nothing cached yet — in
    either case the caller should proceed and do the work normally.
    """
    if key is None:
        return None
    existing = (
        db.query(models.IdempotencyKey)
        .filter(
            models.IdempotencyKey.user_id == user_id,
            models.IdempotencyKey.key == key,
            models.IdempotencyKey.endpoint == endpoint,
        )
        .first()
    )
    if existing is None:
        return None
    return json.loads(existing.response_body)


def _stage_idempotency_record(
    db: Session, user_id: int, key: str | None, endpoint: str, response_body: dict
) -> None:
    """
    Adds (but does NOT commit) an IdempotencyKey row to the session. Callers
    must call this BEFORE their own db.commit(), so the cached response and
    the ledger rows it describes land in the exact same database transaction
    — either both are saved or neither is, which is what makes the cache
    trustworthy. No-ops if no key was supplied.

    Correctness against concurrent duplicates comes from the account row
    lock (for_update=True) each caller already holds by the time this runs:
    two requests with the same key racing on the same account serialize on
    that lock, so the second one only reaches its own cache lookup
    (_get_cached_idempotent_response) after the first has committed —
    at which point it finds the cached row and returns early instead of
    reaching this function at all.
    """
    if key is None:
        return
    record = models.IdempotencyKey(
        user_id=user_id,
        key=key,
        endpoint=endpoint,
        response_body=json.dumps(response_body, default=str),
        status_code=200,
    )
    db.add(record)


@app.post("/accounts/{account_id}/deposit", response_model=schemas.TransactionOut)
def deposit(
    account_id: int,
    body: schemas.DepositRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
    idempotency_key: str | None = Depends(get_idempotency_key),
):
    # Deposit doesn't need for_update for balance-correctness (it always
    # writes unconditionally, never reads the balance to decide anything).
    # It's still taken here, though: it's what serializes two concurrent
    # requests carrying the *same* idempotency key onto the same account, so
    # the second one reliably sees the first one's cached response instead
    # of racing it. See _stage_idempotency_record's docstring.
    account = _get_owned_account(account_id, current_user, db, for_update=True)

    try:
        cached = _get_cached_idempotent_response(db, current_user.id, idempotency_key, "deposit")
        if cached is not None:
            return cached

        assessment = risk.score_transaction(db, account, body.amount)
        txn = models.Transaction(
            account_id=account.id,
            type=models.TransactionType.deposit,
            amount=body.amount,  # positive
            status=assessment.status,
            risk_score=assessment.score,
            risk_reasons=assessment.reasons_text,
        )
        db.add(txn)
        db.flush()  # assigns txn.id without committing, so it can go in the cached response below
        response = schemas.TransactionOut.model_validate(txn).model_dump(mode="json")
        _stage_idempotency_record(db, current_user.id, idempotency_key, "deposit", response)

        db.commit()
        db.refresh(txn)
        return txn
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        logger.exception("Deposit failed for account %s", account_id)
        raise HTTPException(status_code=500, detail="Transaction failed")


@app.post("/accounts/{account_id}/withdraw", response_model=schemas.TransactionOut)
def withdraw(
    account_id: int,
    body: schemas.WithdrawRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
    idempotency_key: str | None = Depends(get_idempotency_key),
):
    # for_update=True: this endpoint reads the balance, then conditionally
    # writes based on it, which is exactly the pattern that needs locking —
    # see the docstring on _get_owned_account. It's also what makes the
    # idempotency check below race-safe (see _stage_idempotency_record).
    account = _get_owned_account(account_id, current_user, db, for_update=True)

    try:
        cached = _get_cached_idempotent_response(db, current_user.id, idempotency_key, "withdraw")
        if cached is not None:
            return cached

        if account.balance < body.amount:
            raise HTTPException(status_code=400, detail="Insufficient balance")

        assessment = risk.score_transaction(db, account, body.amount)
        txn = models.Transaction(
            account_id=account.id,
            type=models.TransactionType.withdrawal,
            amount=-body.amount,  # negative: this is a debit
            status=assessment.status,
            risk_score=assessment.score,
            risk_reasons=assessment.reasons_text,
        )
        db.add(txn)
        db.flush()
        response = schemas.TransactionOut.model_validate(txn).model_dump(mode="json")
        _stage_idempotency_record(db, current_user.id, idempotency_key, "withdraw", response)

        db.commit()
        db.refresh(txn)
        return txn
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        logger.exception("Withdraw failed for account %s", account_id)
        raise HTTPException(status_code=500, detail="Transaction failed")


def _lock_accounts_in_order(account_ids: set[int], db: Session) -> dict[int, models.Account]:
    """
    Lock a set of accounts, always in ascending-id order, regardless of
    which one is logically "from" and which is "to".

    This matters once a single request can lock *two* rows: if transfer
    A->B locks A then B, while a concurrent transfer B->A locks B then A,
    each can end up holding the lock the other is waiting for — a
    deadlock. Postgres detects this and kills one transaction with an
    error rather than hanging forever, but that's still a transfer that
    randomly fails for no reason visible to the caller. Locking in a
    single global order (lowest id first, always) makes that interleaving
    impossible: whichever transfer asks for the lower-id account's lock
    first simply goes first, and the other one waits its turn.

    Returns a dict of {account_id: Account} for whichever of the requested
    ids actually exist — callers check for missing ids themselves, since
    "not found" and "found but not yours" get different status codes.
    """
    locked = {}
    for account_id in sorted(account_ids):
        account = db.query(models.Account).filter(models.Account.id == account_id).with_for_update().first()
        if account:
            locked[account_id] = account
    return locked


@app.post("/accounts/{account_id}/transfer", response_model=schemas.TransferOut)
def transfer(
    account_id: int,
    body: schemas.TransferRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
    idempotency_key: str | None = Depends(get_idempotency_key),
):
    if account_id == body.to_account_id:
        raise HTTPException(status_code=400, detail="Cannot transfer to the same account")

    accounts = _lock_accounts_in_order({account_id, body.to_account_id}, db)

    from_account = accounts.get(account_id)
    if not from_account:
        raise HTTPException(status_code=404, detail="Account not found")
    if from_account.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your account")

    # Locks on both accounts are held by this point (see _lock_accounts_in_order),
    # which is what makes this check race-safe against a concurrent duplicate
    # request carrying the same key — same reasoning as deposit/withdraw.
    try:
        cached = _get_cached_idempotent_response(db, current_user.id, idempotency_key, "transfer")
        if cached is not None:
            return cached

        to_account = accounts.get(body.to_account_id)
        if not to_account:
            raise HTTPException(status_code=404, detail="Destination account not found")

        # Note: no check that to_account belongs to current_user — transfers to
        # other people's accounts are the whole point (that's what makes it a
        # transfer and not just an internal move between your own accounts).

        if from_account.balance < body.amount:
            raise HTTPException(status_code=400, detail="Insufficient balance")

        # One assessment, scored against the sending account, applied to both
        # legs — a transfer is one logical event, so it gets one risk outcome,
        # not two independently-decided ones.
        assessment = risk.score_transaction(db, from_account, body.amount)

        # Two linked rows, same DB transaction, same commit — either both land
        # or neither does. This is the "atomic" part: there's no window where
        # only the debit or only the credit has happened, because both INSERTs
        # are staged against the same session and only become durable at the
        # single db.commit() below. If anything raises before that commit, the
        # whole transaction rolls back and neither row is written.
        debit = models.Transaction(
            account_id=from_account.id,
            type=models.TransactionType.transfer_out,
            amount=-body.amount,
            status=assessment.status,
            risk_score=assessment.score,
            risk_reasons=assessment.reasons_text,
            counterparty_account_id=to_account.id,
        )
        credit = models.Transaction(
            account_id=to_account.id,
            type=models.TransactionType.transfer_in,
            amount=body.amount,
            status=assessment.status,
            risk_score=assessment.score,
            risk_reasons=assessment.reasons_text,
            counterparty_account_id=from_account.id,
        )
        db.add_all([debit, credit])
        db.flush()  # assigns ids to debit/credit without committing

        # Link the two legs so a later review of either one can resolve both
        # together (see /transactions/{id}/review) instead of leaving a
        # transfer half-approved.
        debit.related_transaction_id = credit.id
        credit.related_transaction_id = debit.id

        response = schemas.TransferOut(debit=debit, credit=credit).model_dump(mode="json")
        _stage_idempotency_record(db, current_user.id, idempotency_key, "transfer", response)

        db.commit()
        db.refresh(debit)
        db.refresh(credit)

        return schemas.TransferOut(debit=debit, credit=credit)
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        logger.exception("Transfer failed from account %s to %s", account_id, body.to_account_id)
        raise HTTPException(status_code=500, detail="Transaction failed")


@app.get("/transactions/{transaction_id}", response_model=schemas.TransactionOut)
def get_transaction(
    transaction_id: int,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    txn = db.query(models.Transaction).filter(models.Transaction.id == transaction_id).first()
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")

    # A transaction doesn't know its owner directly — it belongs to an
    # account, which belongs to a user. Walk that chain to check access,
    # same as _get_owned_account does for the account-scoped endpoints.
    account = db.query(models.Account).filter(models.Account.id == txn.account_id).first()
    if not account or account.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your transaction")

    return txn


@app.get("/accounts/{account_id}/transactions", response_model=list[schemas.TransactionOut])
def get_transactions(
    account_id: int,
    limit: int = Query(10, ge=1, le=100, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Rows to skip"),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    account = _get_owned_account(account_id, current_user, db)
    return (
        db.query(models.Transaction)
        .filter(models.Transaction.account_id == account.id)
        .order_by(models.Transaction.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def require_admin(current_user: models.User = Depends(auth.get_current_user)) -> models.User:
    """
    There's deliberately no signup flag or endpoint that grants is_admin —
    see the comment on User.is_admin in models.py. Locally, promote a user
    with `python scripts/make_admin.py <email>`.
    """
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


@app.get("/admin/transactions/flagged", response_model=list[schemas.TransactionOut])
def list_flagged_transactions(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _admin: models.User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return (
        db.query(models.Transaction)
        .filter(models.Transaction.status == models.TransactionStatus.flagged_for_review)
        .order_by(models.Transaction.created_at.asc())  # oldest-first: review queues are FIFO
        .offset(offset)
        .limit(limit)
        .all()
    )


@app.post("/transactions/{transaction_id}/review", response_model=schemas.TransactionOut)
def review_transaction(
    transaction_id: int,
    body: schemas.ReviewDecision,
    admin: models.User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    txn = (
        db.query(models.Transaction)
        .filter(models.Transaction.id == transaction_id)
        .with_for_update()
        .first()
    )
    if not txn:
        raise HTTPException(status_code=404, detail="Transaction not found")
    if txn.status != models.TransactionStatus.flagged_for_review:
        raise HTTPException(
            status_code=400,
            detail=f"Transaction is '{txn.status.value}', not pending review",
        )

    # If this is one leg of a transfer, lock and resolve its sibling in the
    # same commit — approving only one side would leave the transfer half
    # done, exactly the atomicity problem Week 2 exists to prevent. This is
    # the same "lock both, act on both, one commit" shape as the transfer
    # endpoint itself.
    sibling = None
    if txn.related_transaction_id is not None:
        sibling = (
            db.query(models.Transaction)
            .filter(models.Transaction.id == txn.related_transaction_id)
            .with_for_update()
            .first()
        )

    new_status = (
        models.TransactionStatus.approved
        if body.decision == "approve"
        else models.TransactionStatus.blocked
    )
    txn.status = new_status
    if sibling is not None:
        sibling.status = new_status
    if body.note:
        note = f"[reviewed by admin user {admin.id}: {body.note}]"
        txn.risk_reasons = f"{txn.risk_reasons}; {note}" if txn.risk_reasons else note

    db.commit()
    db.refresh(txn)
    return txn

@app.get("/")
def root():
    return {"message": "Digital Wallet API is running", "docs": "/docs"}

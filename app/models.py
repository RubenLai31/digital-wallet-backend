"""
Database models.

Design note: this uses a "ledger" pattern rather than just an `accounts.balance`
column you update in place. Every deposit, withdrawal, or transfer creates one
or more Transaction rows, and an account's balance is the SUM of its
transactions. This is how real financial systems work, for a reason:

- You get a full audit trail for free (every balance change is a row you can
  inspect later — required for any real fintech compliance work)
- It's much harder to "lose" money to a race condition, because you're always
  INSERTing new rows, never overwriting a balance in place
- A transfer between two accounts is two linked Transaction rows (one debit,
  one credit) that must succeed or fail together (see transactions.py)
"""

from datetime import datetime, timezone
from decimal import Decimal
from sqlalchemy import (
    Column,
    Integer,
    String,
    Numeric,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Enum,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
import enum

from .database import Base


class TransactionType(str, enum.Enum):
    deposit = "deposit"
    withdrawal = "withdrawal"
    transfer_out = "transfer_out"
    transfer_in = "transfer_in"


class TransactionStatus(str, enum.Enum):
    approved = "approved"
    flagged_for_review = "flagged_for_review"
    blocked = "blocked"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    # Never settable via the public API (see UserCreate in schemas.py) —
    # only promotable through the scripts/make_admin.py dev tool. A signup
    # endpoint that could mint admins would defeat the point of having a
    # reviewer role at all.
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    accounts = relationship("Account", back_populates="owner")


class Account(Base):
    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    currency = Column(String, default="USD")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    owner = relationship("User", back_populates="accounts")
    transactions = relationship(
        "Transaction", back_populates="account", foreign_keys="Transaction.account_id"
    )

    @property
    def balance(self) -> Decimal:
        """Balance is derived, never stored directly — see module docstring."""
        return sum(
            (t.amount for t in self.transactions if t.status == TransactionStatus.approved),
            Decimal("0.00"),
        )


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)
    type = Column(Enum(TransactionType), nullable=False)
    amount = Column(Numeric(12, 2), nullable=False)  # positive = credit, negative = debit
    status = Column(Enum(TransactionStatus), default=TransactionStatus.approved)
    risk_score = Column(Float, nullable=True)  # set by app/risk.py on every deposit/withdraw/transfer
    risk_reasons = Column(String, nullable=True)  # human-readable, comma-joined; None if score was 0
    counterparty_account_id = Column(Integer, ForeignKey("accounts.id"), nullable=True)
    # For a transfer's debit/credit pair only — points each row at its
    # sibling, so reviewing a flagged transfer can resolve both legs in one
    # commit instead of leaving one approved and the other stuck pending.
    # Null for deposits and withdrawals, which have no sibling.
    related_transaction_id = Column(Integer, ForeignKey("transactions.id"), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    account = relationship("Account", back_populates="transactions", foreign_keys=[account_id])


class IdempotencyKey(Base):
    """
    Caches the response of a money-moving request against a client-supplied
    key, so a retried request (network blip, double-click, client retry
    logic) returns the *same* result instead of moving money twice.

    Scoped to (user_id, key, endpoint): the same key string is allowed to
    mean different things on different endpoints, and can't be reused by a
    different user to read another user's cached response.

    response_body stores the JSON-serialized API response (built from the
    same Pydantic schema the endpoint normally returns), so a repeated
    request can be answered without recomputing anything.
    """

    __tablename__ = "idempotency_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key = Column(String, nullable=False)
    endpoint = Column(String, nullable=False)
    response_body = Column(String, nullable=False)  # JSON-encoded response
    status_code = Column(Integer, nullable=False, default=200)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        UniqueConstraint("user_id", "key", "endpoint", name="uq_idempotency_user_key_endpoint"),
    )


class RefreshToken(Base):
    """
    Backs the refresh-token flow (see auth.py). Access tokens are stateless
    JWTs and deliberately short-lived (15 min) — they can't be revoked
    early, so keeping them short bounds how long a leaked one is dangerous
    for. Refresh tokens are the opposite: long-lived (7 days) but tracked
    here in the database specifically so they *can* be revoked on logout,
    or rotated on use.

    token_hash stores a SHA-256 hash of the actual token, never the token
    itself — same reasoning as password hashing: if this table ever leaks,
    an attacker holding only the hash can't reconstruct a usable token from
    it. The raw token is returned to the client once, at issue time, and
    never stored anywhere in plaintext.
    """

    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token_hash = Column(String, unique=True, index=True, nullable=False)
    revoked = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime, nullable=False)
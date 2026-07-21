"""
Fraud / risk scoring for money-moving transactions.

Every deposit, withdrawal, and transfer gets scored 0.0-1.0 by a handful of
heuristics before it's recorded, and the score decides its status:

    score <  FLAG_THRESHOLD   -> approved            (counts toward balance immediately)
    score <  BLOCK_THRESHOLD  -> flagged_for_review   (recorded, but held out of the
                                                         balance until a human reviews it)
    score >= BLOCK_THRESHOLD  -> blocked              (recorded, rejected outright)

Notice that "held out of the balance" isn't new machinery — Account.balance
already only sums transactions with status == approved (see models.py).
Flagging or blocking a transaction is entirely a matter of *not* setting
status to approved yet; nothing else needs to change to make that "held
money" behavior correct. The transaction row is always created regardless
of outcome, too — same ledger-pattern reasoning as everywhere else in this
app: even a blocked attempt is worth an audit trail, not a silently
swallowed request.

Real fraud systems train models on historical labeled fraud data and pull
in device fingerprints, IP geolocation, and dozens of other signals that
update continuously. This is a hand-written heuristic scorer instead: good
enough to learn the *shape* of the problem — score a transaction, hold
suspicious ones out of the ledger, resolve them via a review step, and test
all three outcomes deterministically — without needing a training pipeline
or a labeled dataset this project doesn't have.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy.orm import Session

from . import models

# --- Amount: a transaction much larger than "normal" is worth a second look ---
LARGE_AMOUNT_THRESHOLD = Decimal("10000")
LARGE_AMOUNT_SCORE = 0.5
MEDIUM_AMOUNT_THRESHOLD = Decimal("1000")
MEDIUM_AMOUNT_SCORE = 0.2

# --- Velocity: many transactions on one account in a short window ---
# (a classic "someone has your credentials and is draining the account" signal)
VELOCITY_WINDOW_MINUTES = 10
VELOCITY_COUNT_THRESHOLD = 5
VELOCITY_SCORE = 0.4

# --- New account: a brand-new account moving a meaningful amount right away ---
# (a classic money-mule / stolen-card pattern: open account, move money, vanish)
NEW_ACCOUNT_MINUTES = 5
NEW_ACCOUNT_AMOUNT_THRESHOLD = Decimal("500")
NEW_ACCOUNT_SCORE = 0.4

FLAG_THRESHOLD = 0.4
BLOCK_THRESHOLD = 0.8


def _as_utc(dt: datetime) -> datetime:
    """
    Same reasoning as auth._as_utc: SQLite can hand back a naive datetime
    for a value that was stored timezone-aware. Every datetime this app
    stores is UTC by convention, so a naive one is always safe to treat as
    UTC rather than local time.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class RiskAssessment:
    score: float
    status: models.TransactionStatus
    reasons: list[str] = field(default_factory=list)

    @property
    def reasons_text(self) -> str | None:
        """What actually gets stored in Transaction.risk_reasons — None if score was 0."""
        return ", ".join(self.reasons) if self.reasons else None


def score_transaction(db: Session, account: models.Account, amount: Decimal) -> RiskAssessment:
    """
    Score a single money-moving amount against `account`'s recent activity.

    `amount` should be the magnitude being moved — a deposit's amount, a
    withdrawal's or transfer debit's absolute value, not its signed ledger
    value (a -500 withdrawal and a +500 deposit should score identically;
    the sign is a bookkeeping detail, not a risk signal here).
    """
    score = 0.0
    reasons: list[str] = []

    if amount >= LARGE_AMOUNT_THRESHOLD:
        score += LARGE_AMOUNT_SCORE
        reasons.append(f"amount >= {LARGE_AMOUNT_THRESHOLD}")
    elif amount >= MEDIUM_AMOUNT_THRESHOLD:
        score += MEDIUM_AMOUNT_SCORE
        reasons.append(f"amount >= {MEDIUM_AMOUNT_THRESHOLD}")

    window_start = datetime.now(timezone.utc) - timedelta(minutes=VELOCITY_WINDOW_MINUTES)
    recent_count = (
        db.query(models.Transaction)
        .filter(
            models.Transaction.account_id == account.id,
            models.Transaction.created_at >= window_start,
        )
        .count()
    )
    if recent_count >= VELOCITY_COUNT_THRESHOLD:
        score += VELOCITY_SCORE
        reasons.append(f"{recent_count} transactions in the last {VELOCITY_WINDOW_MINUTES}m")

    account_age = datetime.now(timezone.utc) - _as_utc(account.created_at)
    if account_age < timedelta(minutes=NEW_ACCOUNT_MINUTES) and amount >= NEW_ACCOUNT_AMOUNT_THRESHOLD:
        score += NEW_ACCOUNT_SCORE
        reasons.append(
            f"account age {account_age} < {NEW_ACCOUNT_MINUTES}m "
            f"and amount >= {NEW_ACCOUNT_AMOUNT_THRESHOLD}"
        )

    score = min(score, 1.0)

    if score >= BLOCK_THRESHOLD:
        status = models.TransactionStatus.blocked
    elif score >= FLAG_THRESHOLD:
        status = models.TransactionStatus.flagged_for_review
    else:
        status = models.TransactionStatus.approved

    return RiskAssessment(score=score, status=status, reasons=reasons)
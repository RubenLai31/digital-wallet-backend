"""
Pydantic schemas — define the shape of API requests/responses.
Kept separate from database models (app/models.py) on purpose: the API's
public shape and the database's internal shape are allowed to diverge, and
usually should (e.g. we never want to accidentally return hashed_password).
"""

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

# Money as Decimal, not float — float can't exactly represent most decimal
# fractions (0.1 + 0.2 != 0.3 in binary floating point), which is exactly
# the kind of rounding error you can't have in a ledger. Two flavors:
# - Money: any value a stored amount/balance can take (can be negative — a
#   withdrawal is a negative amount; a balance could in principle go there).
# - PositiveMoney: what a deposit/withdraw/transfer *request* must contain.
Money = Annotated[Decimal, Field(max_digits=12, decimal_places=2)]
PositiveMoney = Annotated[Decimal, Field(gt=0, max_digits=12, decimal_places=2)]


class UserCreate(BaseModel):
    email: EmailStr
    # max_length guards against bcrypt's 72-byte limit: passwords longer than
    # that are silently truncated by bcrypt, so two different passwords
    # sharing the same first 72 bytes would hash identically. Better to
    # reject upfront than have that footgun. (72 chars is a fast pre-check;
    # the validator below checks actual UTF-8 byte length, since multi-byte
    # characters can hit 72 bytes well before 72 characters.)
    password: str = Field(min_length=8, max_length=72)

    @field_validator("password")
    @classmethod
    def password_fits_bcrypt(cls, v: str) -> str:
        if len(v.encode("utf-8")) > 72:
            raise ValueError("Password must be 72 bytes or fewer (bcrypt limitation)")
        return v


class UserOut(BaseModel):
    id: int
    email: EmailStr
    is_admin: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AccountOut(BaseModel):
    id: int
    owner_id: int
    currency: str
    balance: Money

    model_config = ConfigDict(from_attributes=True)


class DepositRequest(BaseModel):
    amount: PositiveMoney = Field(description="Must be positive")


class WithdrawRequest(BaseModel):
    amount: PositiveMoney = Field(description="Must be positive")


class TransferRequest(BaseModel):
    to_account_id: int = Field(gt=0, description="Must be a positive account ID")
    amount: PositiveMoney = Field(description="Must be positive")


class TransactionOut(BaseModel):
    id: int
    account_id: int
    type: str
    amount: Money
    status: str
    risk_score: float | None
    risk_reasons: str | None
    related_transaction_id: int | None
    counterparty_account_id: int | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TransferOut(BaseModel):
    """A transfer produces two linked transactions — show both sides."""

    debit: TransactionOut
    credit: TransactionOut


class ReviewDecision(BaseModel):
    decision: Literal["approve", "reject"]
    note: str | None = Field(default=None, max_length=500)


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str
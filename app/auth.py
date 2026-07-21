"""
Authentication: password hashing + JWT token issuing/verification.

This is Week 2 scope, but it's included from day one since account creation
naturally needs a password. Feel free to skip actually enforcing auth on
endpoints until Week 2 — the pieces are here when you're ready.
"""

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from .database import get_db
from . import models

load_dotenv()  # reads .env in the project root into os.environ, if present

# Never hardcode this — load from an environment variable or secrets manager.
# Missing at startup is a hard failure: running with a known secret lets
# anyone forge valid JWTs for any user (including admins). Local dev should
# set this in a .env file (see README).
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "JWT_SECRET_KEY environment variable is not set. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\" "
        "and add it to your .env file."
    )
ALGORITHM = "HS256"
# Short-lived on purpose: access tokens are stateless JWTs, so there's no
# way to revoke one early if it leaks. Keeping the window short bounds the
# damage; the refresh-token flow below is what lets a client stay logged in
# past 15 minutes without re-entering a password.
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    # jti (JWT ID): a random per-token identifier. Without this, two tokens
    # minted for the same user within the same second (exp has 1-second
    # resolution) would be byte-for-byte identical — harmless for auth
    # itself, but it broke the "refreshing gives you a genuinely new token"
    # guarantee, which the test suite caught.
    to_encode.update({"exp": expire, "jti": secrets.token_hex(8)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> models.User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        user_id = int(user_id)  # a malformed/non-numeric sub claim -> ValueError, not 500
    except (JWTError, ValueError):
        raise credentials_exception

    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user is None:
        raise credentials_exception
    return user


def _hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _as_utc(dt: datetime) -> datetime:
    """
    SQLite (unlike Postgres) doesn't reliably preserve tzinfo on round-trip
    through a plain DateTime column — a value stored as timezone-aware can
    come back naive. Comparing a naive and an aware datetime raises
    TypeError, so every comparison against "now" goes through this first.
    Every datetime this app stores is UTC by convention, so treating a
    naive value as UTC (rather than local time) is always correct here.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def issue_refresh_token(db: Session, user_id: int) -> str:
    """
    Creates a new refresh token, stores its hash, and returns the raw token
    to hand to the client. The raw value is never persisted — only its hash
    (see RefreshToken.token_hash docstring in models.py).
    """
    raw_token = secrets.token_urlsafe(32)
    record = models.RefreshToken(
        user_id=user_id,
        token_hash=_hash_token(raw_token),
        expires_at=datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(record)
    db.commit()
    return raw_token


def get_valid_refresh_token(db: Session, raw_token: str) -> models.RefreshToken | None:
    """Returns the matching RefreshToken row iff it exists, isn't revoked, and hasn't expired."""
    record = (
        db.query(models.RefreshToken)
        .filter(models.RefreshToken.token_hash == _hash_token(raw_token))
        .first()
    )
    if record is None or record.revoked:
        return None
    if _as_utc(record.expires_at) < datetime.now(timezone.utc):
        return None
    return record


def revoke_refresh_token(db: Session, raw_token: str) -> None:
    """No-ops silently on an unknown/already-revoked token — logout should never error on that."""
    record = (
        db.query(models.RefreshToken)
        .filter(models.RefreshToken.token_hash == _hash_token(raw_token))
        .first()
    )
    if record is not None and not record.revoked:
        record.revoked = True
        db.commit()
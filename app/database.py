"""
Database connection setup.

Defaults to a local SQLite file so you can run everything with zero setup.
Swap DATABASE_URL to a Postgres connection string (see docker-compose.yml,
added in Week 4) when you're ready for something closer to production —
no other code needs to change.
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()  # must happen before DATABASE_URL is read below

# Render (and some other PaaS providers) hand out DATABASE_URL with a
# `postgres://` scheme. SQLAlchemy 2.x requires `postgresql://` — the old
# short form was removed. Convert it here so the app works on Render without
# the user having to manually edit the connection string.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./wallet.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


def make_engine(database_url: str):
    """
    Factory (not just a bare module-level engine) so tests can build their
    own SQLite engines with the same locking behavior as the real app —
    see the note below on why that behavior needs to be opted into
    explicitly for SQLite.
    """
    is_sqlite = database_url.startswith("sqlite")
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    new_engine = create_engine(database_url, connect_args=connect_args)

    if is_sqlite:
        # SQLite has no real row-level locking, and by default starts
        # transactions in "deferred" mode: a read doesn't take any lock at
        # all, so two concurrent transactions can both read the same row
        # before either has written anything — exactly the race that
        # `with_for_update()` (used in main.py's withdraw endpoint) is
        # supposed to prevent. On Postgres, with_for_update() works out of
        # the box because Postgres has real row locks; SQLite needs to be
        # told to acquire a write lock immediately at BEGIN, before any
        # reads happen, or with_for_update() is a no-op. This is the
        # standard SQLAlchemy-recommended workaround for that gap:
        # https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#serializable-isolation-savepoints-transactional-ddl
        @event.listens_for(new_engine, "connect")
        def _set_sqlite_isolation(dbapi_connection, connection_record):
            # Stop pysqlite from emitting its own implicit BEGIN, so we can
            # control exactly when and how transactions start below.
            dbapi_connection.isolation_level = None

        @event.listens_for(new_engine, "begin")
        def _begin_immediate(conn):
            conn.exec_driver_sql("BEGIN IMMEDIATE")

    return new_engine


engine = make_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a DB session per request, closes it after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
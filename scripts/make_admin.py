"""
Dev utility: promote a user to admin by email, so they can review flagged
transactions (see /admin/transactions/flagged and /transactions/{id}/review
in app/main.py).

There's deliberately no API endpoint for this. A signup flag or an
in-app "become admin" call would let anyone grant themselves reviewer
access, which defeats the entire point of having a review step — the
fraud-flagging system exists precisely so the account owner *can't* just
wave their own flagged transaction through. In a real system this would be
an internal-only tool with its own separate access control, not something
reachable from the public API; this script is the local-dev equivalent of
that split.

Usage (run from the project root, with your venv active):
  python scripts/make_admin.py you@example.com
"""

import sys
from pathlib import Path

# So `app` is importable when this is run as `python scripts/make_admin.py`
# from the project root, rather than as an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal  # noqa: E402
from app import models  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/make_admin.py <email>")
        sys.exit(1)

    email = sys.argv[1]
    db = SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.email == email).first()
        if user is None:
            print(f"No user found with email {email!r}")
            sys.exit(1)
        if user.is_admin:
            print(f"{email} is already an admin.")
            return
        user.is_admin = True
        db.commit()
        print(f"{email} is now an admin.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
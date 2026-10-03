"""Time helpers.

``datetime.utcnow()`` is deprecated (Python 3.12+) and returns a naive
datetime. The codebase stores and compares *naive UTC* datetimes throughout
(pydantic model defaults, ``isoformat()`` strings persisted to SQLite,
subtraction between timestamps), so this helper preserves exactly that
behaviour while deriving the value from a timezone-aware clock.
"""

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Return the current UTC time as a naive datetime (same as the old ``datetime.utcnow()``)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)  # noqa: UP017 - datetime.UTC needs Python 3.11; CI also tests 3.10

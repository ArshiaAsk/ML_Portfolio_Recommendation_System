"""Date utility helpers."""

from datetime import date, datetime, timezone
from typing import Optional


def today_utc() -> date:
    """Return today's date in UTC."""
    return datetime.now(tz=timezone.utc).date()


def now_utc() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(tz=timezone.utc)


def resolve_end_date(end_date: Optional[str]) -> date:
    """Resolve end_date from config: return today if None or 'null'.

    Args:
        end_date: ISO date string or None/``"null"``.

    Returns:
        Resolved :class:`datetime.date`.
    """
    if end_date is None or str(end_date).lower() in ("null", "none", ""):
        return today_utc()
    return date.fromisoformat(str(end_date))


def parse_date(value: str) -> date:
    """Parse an ISO-8601 date string into a :class:`datetime.date`.

    Args:
        value: ISO date string, e.g. ``"2018-01-01"``.

    Returns:
        :class:`datetime.date`.
    """
    return date.fromisoformat(value)

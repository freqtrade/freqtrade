from datetime import datetime, timezone
from typing import Optional


def dt_now() -> datetime:
    """Return the current datetime in UTC."""
    return datetime.now(timezone.utc)


def dt_ts(dt: Optional[datetime] = None) -> int:
    """Return the current timestamp in ms as a timestamp in UTC."""
    if dt:
        return int(dt.timestamp() * 1000)
    return int(dt_now().timestamp() * 1000)


def dt_from_ts(timestamp: float) -> datetime:
    """
    Return a datetime from a timestamp.
    :param timestamp: timestamp in seconds or milliseconds
    """
    if timestamp > 1e10:
        # Timezone in ms - convert to seconds
        timestamp /= 1000
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)

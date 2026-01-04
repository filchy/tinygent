from datetime import datetime
from datetime import timezone
from uuid import uuid4

from neo4j import time as neo4j_time


def generate_uuid() -> str:
    return str(uuid4())


def ensure_utc(dt: datetime | None) -> datetime | None:
    if not dt:
        return None

    if dt.tzinfo is None:
        # If datetime is naive, assume it's UTC
        return dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo != timezone.utc:
        # If datetime has a different timezone, convert to UTC
        return dt.astimezone(timezone.utc)

    return dt


def get_current_timestamp() -> datetime:
    return datetime.now(timezone.utc)


def parse_timestamp(dt: datetime) -> str:
    dt = ensure_utc(dt) or get_current_timestamp()
    return dt.isoformat(timespec='microseconds').replace('+00:00', 'Z')


def parse_db_date(input_date: neo4j_time.DateTime | str) -> datetime:
    if isinstance(input_date, neo4j_time.DateTime):
        return input_date.to_native()

    if isinstance(input_date, str):
        return datetime.fromisoformat(input_date)

    raise ValueError(f'Unsupported input date: {type(input_date)}')


def get_default_subgraph_id() -> str:
    return ''

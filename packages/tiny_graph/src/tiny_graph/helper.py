from datetime import datetime
from datetime import timezone
from uuid import uuid4
from neo4j import time as neo4j_time


def generate_uuid() -> str:
    return str(uuid4())


def get_current_timestamp() -> datetime:
    return datetime.now(timezone.utc)


def parse_db_date(input_date: neo4j_time.DateTime | str) -> datetime:
    if isinstance(input_date, neo4j_time.DateTime):
        return input_date.to_native()

    if isinstance(input_date, str):
        return datetime.fromisoformat(input_date)

    raise ValueError(f'Unsupported input date: {type(input_date)}')


def get_default_subgraph_id() -> str:
    return ''

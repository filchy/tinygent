import inspect
from typing import Any
from typing import Callable

from tiny_chat.message import BaseMessage

_message_fn = None


def on_message(fn: Callable[[BaseMessage], Any]):
    global _message_fn
    _message_fn = fn
    return fn


async def call_message(msg: BaseMessage):
    if _message_fn:
        if inspect.iscoroutinefunction(_message_fn):
            return await _message_fn(msg)
        return _message_fn(msg)
    return None

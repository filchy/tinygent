import inspect

from tiny_chat.message import UserMessage


_message_fn = None


def on_message(fn):
    global _message_fn
    _message_fn = fn
    return fn


async def call_message(msg: UserMessage):
    if _message_fn:
        if inspect.iscoroutinefunction(_message_fn):
            return await _message_fn(msg)
        return _message_fn(msg)
    return 'No handler registered.'

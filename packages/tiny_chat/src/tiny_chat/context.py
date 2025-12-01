from __future__ import annotations

from contextvars import ContextVar
import typing

if typing.TYPE_CHECKING:
    from tiny_chat.session import WebsocketSession

current_chat_id: ContextVar[str] = ContextVar('current_chat_id')
current_session: ContextVar[WebsocketSession] = ContextVar('current_session')

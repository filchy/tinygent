from asyncio import Task
from typing import Any

from fastapi import WebSocket

from tiny_chat.context import current_session
from tiny_chat.emitter import emitter
from tiny_chat.message import BaseMessage


class SessionStore:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.store.get(key, default)

    def has(self, key: str) -> bool:
        return key in self.store


class BaseSession(SessionStore):
    def __init__(self, id: str, user: str | None = None):
        super().__init__()

        self.session_id = id
        self.user = user


class WebsocketSession(BaseSession):
    def __init__(self, session_id: str, socket_id: str, ws: WebSocket):
        super().__init__(session_id)
        self.socket_id = socket_id
        self.ws = ws
        self.active_task: Task | None = None

        emitter.configure(self._send_json)

        ws_sessions_sid[socket_id] = self
        ws_sessions_id[session_id] = self

        current_session.set(self)

    async def _send_json(self, msg: BaseMessage) -> None:
        await self.ws.send_json(msg.model_dump())

    def restore(self, new_socket_id: str, new_ws: WebSocket) -> None:
        ws_sessions_sid.pop(self.socket_id, None)
        ws_sessions_sid[new_socket_id] = self
        self.socket_id = new_socket_id
        self.ws = new_ws


ws_sessions_sid: dict[str, WebsocketSession] = {}
ws_sessions_id: dict[str, WebsocketSession] = {}

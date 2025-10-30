from fastapi import WebSocket

from tiny_chat.emitter import emitter
from tiny_chat.message import BaseMessage


class BaseSession:
    def __init__(self, id: str, user: str | None = None):
        self.id = id
        self.user = user
        self.history: list[BaseMessage] = []


class WebsocketSession(BaseSession):
    def __init__(self, id: str, socket_id: str, ws: WebSocket):
        super().__init__(id)
        self.socket_id = socket_id
        self.ws = ws

        emitter.configure(self._send_json)

        ws_sessions_sid[socket_id] = self
        ws_sessions_id[id] = self

    async def _send_json(self, msg: BaseMessage):
        await self.ws.send_json(msg.model_dump())

    def restore(self, new_socket_id: str, new_ws: WebSocket):
        ws_sessions_sid.pop(self.socket_id, None)
        ws_sessions_sid[new_socket_id] = self
        self.socket_id = new_socket_id
        self.ws = new_ws


ws_sessions_sid: dict[str, WebsocketSession] = {}
ws_sessions_id: dict[str, WebsocketSession] = {}

from fastapi import WebSocket

from tiny_chat.message import Message


class BaseSession:
    def __init__(self, id: str, user: str | None = None):
        self.id = id
        self.user = user
        self.history: list[Message] = []


class WebsocketSession(BaseSession):
    def __init__(self, id: str, socket_id: str, ws: WebSocket):
        super().__init__(id)
        self.socket_id = socket_id
        self.ws = ws

        ws_sessions_sid[socket_id] = self
        ws_sessions_id[id] = self

    async def emit(self, message: Message):
        await self.ws.send_json(message.model_dump_json())

    def restore(self, new_socket_id: str, new_ws: WebSocket):
        ws_sessions_sid.pop(self.socket_id, None)
        ws_sessions_sid[new_socket_id] = self
        self.socket_id = new_socket_id
        self.ws = new_ws


ws_sessions_sid: dict[str, WebsocketSession] = {}
ws_sessions_id: dict[str, WebsocketSession] = {}

from asyncio import Task
from collections import defaultdict

from fastapi import WebSocket

from tiny_chat.context import current_chat_id
from tiny_chat.emitter import emitter
from tiny_chat.message import AgentMessage
from tiny_chat.message import AgentMessageChunk
from tiny_chat.message import BaseMessage
from tiny_chat.message import MessageUnion


class BaseSession:
    def __init__(self, id: str, user: str | None = None):
        self.session_id = id
        self.user = user
        self.chats: dict[str, list[BaseMessage]] = defaultdict(list)

    @property
    def clean_chat(self) -> list[BaseMessage]:
        msgs = sorted(self.chats[current_chat_id.get()], key=lambda m: m._created_at)

        clean: list[BaseMessage] = []
        chunks: dict[str, list[AgentMessageChunk]] = defaultdict(list)

        for m in msgs:
            if isinstance(m, AgentMessageChunk):
                chunks[m.id].append(m)

        processed_ids: set[str] = set()

        for m in msgs:
            if isinstance(m, AgentMessageChunk):
                if m.id in processed_ids:
                    continue
                processed_ids.add(m.id)

                group = chunks[m.id]
                merged = AgentMessage(
                    id=m.id, content=''.join(chunk.content for chunk in group)
                )
                clean.append(merged)
            else:
                clean.append(m)

        return clean


class WebsocketSession(BaseSession):
    def __init__(self, session_id: str, socket_id: str, ws: WebSocket):
        super().__init__(session_id)
        self.socket_id = socket_id
        self.ws = ws
        self.active_task: Task | None = None

        emitter.configure(self._send_json)

        ws_sessions_sid[socket_id] = self
        ws_sessions_id[session_id] = self

    def _add_message(self, msg: BaseMessage):
        parsed_msg = MessageUnion.validate_python(msg.model_dump())
        self.chats[current_chat_id.get()].append(parsed_msg)

    async def _send_json(self, msg: BaseMessage):
        await self.ws.send_json(msg.model_dump())
        self._add_message(msg)

    def restore(self, new_socket_id: str, new_ws: WebSocket):
        ws_sessions_sid.pop(self.socket_id, None)
        ws_sessions_sid[new_socket_id] = self
        self.socket_id = new_socket_id
        self.ws = new_ws


ws_sessions_sid: dict[str, WebsocketSession] = {}
ws_sessions_id: dict[str, WebsocketSession] = {}

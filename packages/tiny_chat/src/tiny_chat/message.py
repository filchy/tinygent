from datetime import datetime
from typing import Any
from typing import Literal
import uuid

from pydantic import BaseModel
from pydantic import PrivateAttr
from pydantic import TypeAdapter

from tiny_chat.emitter import emitter


class BaseMessage(BaseModel):
    id: str = str(uuid.uuid4())
    type: Any
    sender: Any
    content: str

    _created_at: datetime = PrivateAttr(default_factory=datetime.now)

    async def send(self):
        await emitter.send(self)


class UserMessage(BaseMessage):
    type: Literal['text'] = 'text'
    sender: Literal['user'] = 'user'

    chat_id: str


class AgentMessage(BaseMessage):
    type: Literal['text'] = 'text'
    sender: Literal['agent'] = 'agent'


class AgentMessageChunk(BaseMessage):
    type: Literal['chunk'] = 'chunk'
    sender: Literal['agent'] = 'agent'


class ChildBaseMessage(BaseMessage):
    parent_id: str


class AgentToolCallMessage(ChildBaseMessage):
    type: Literal['tool'] = 'tool'
    sender: Literal['agent'] = 'agent'

    content: Any = ''
    tool_name: str
    tool_args: dict[str, Any]


class AgentSourceMessage(ChildBaseMessage):
    type: Literal['source'] = 'source'
    sender: Literal['agent'] = 'agent'

    content: str = ''
    name: str
    url: str
    favicon: str | None = None
    description: str | None = None


MessageUnion: TypeAdapter[
    UserMessage
    | AgentMessage
    | AgentMessageChunk
    | AgentToolCallMessage
    | AgentSourceMessage
] = TypeAdapter(
    UserMessage
    | AgentMessage
    | AgentMessageChunk
    | AgentToolCallMessage
    | AgentSourceMessage
)

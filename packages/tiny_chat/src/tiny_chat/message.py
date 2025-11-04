from typing import Any
from typing import Literal
import uuid

from pydantic import BaseModel

from tiny_chat.emitter import emitter


class BaseMessage(BaseModel):
    id: str = str(uuid.uuid4())
    type: Any
    sender: Any
    content: str

    async def send(self):
        await emitter.send(self)


class AgentAnswerMessage(BaseMessage):
    type: Literal['text'] = 'text'
    sender: Literal['agent'] = 'agent'


class AgentToolCallMessage(BaseMessage):
    parent_id: str
    type: Literal['tool'] = 'tool'
    sender: Literal['agent'] = 'agent'
    content: str = ''
    tool_name: str
    tool_args: dict[str, Any]

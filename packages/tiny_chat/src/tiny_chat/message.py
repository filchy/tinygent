from typing import Literal
import uuid

from pydantic import BaseModel

from tiny_chat.emitter import emitter


class BaseMessage(BaseModel):
    id: str = str(uuid.uuid4())
    type: Literal['text', 'reasoning', 'debug', 'delta']
    sender: Literal['user', 'agent']
    content: str

    async def send(self):
        await emitter.send(self)

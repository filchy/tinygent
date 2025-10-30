from pydantic import BaseModel
from typing import Literal
from typing import Optional

from tiny_chat.emitter import emitter


class BaseMessage(BaseModel):
    id: str
    type: Literal['text', 'reasoning', 'debug', 'delta']
    sender: Literal['user', 'agent']
    content: str
    streaming: Optional[bool] = False

    async def send(self):
        await emitter.send(self)

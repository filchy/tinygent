from typing import Any
from typing import Generic
from typing import Literal
from typing import TypeVar

from pydantic import BaseModel
from pydantic import ConfigDict

TinyMessageType = TypeVar(
    'TinyMessageType', Literal['chat'], Literal['tool'], Literal['human']
)


class BaseMessage(BaseModel, Generic[TinyMessageType]):
    type: TinyMessageType

    metadata: dict = {}

    model_config = ConfigDict(frozen=True, extra='forbid')


class TinyChatMessage(BaseMessage[Literal['chat']]):
    type: Literal['chat'] = 'chat'

    content: str

    metadata: dict = {}


class TinyToolCall(BaseMessage[Literal['tool']]):
    type: Literal['tool'] = 'tool'

    tool_name: str

    arguments: dict

    call_id: str | None = None

    result: Any | None = None

    metadata: dict = {}


class TinyHumanMessage(BaseMessage[Literal['human']]):
    type: Literal['human'] = 'human'

    content: str

    metadata: dict = {}


TinyAIMessage = TinyChatMessage | TinyToolCall

AllTinyMessages = TinyAIMessage | TinyHumanMessage

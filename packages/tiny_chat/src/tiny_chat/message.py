from pydantic import BaseModel
from typing import Generic
from typing import TypeVar
from typing import Literal
from typing import Optional

TType = TypeVar('TType', bound=str)
TSender = TypeVar('TSender', bound=str)


class BaseMessage(BaseModel, Generic[TType, TSender]):
    id: str
    type: TType
    sender: TSender
    streaming: Optional[bool] = False


class UserMessage(BaseMessage[Literal['text'], Literal['user']]):
    content: str


class AgentTextMessage(BaseMessage[Literal['text'], Literal['agent']]):
    content: str


class AgentReasoningMessage(BaseMessage[Literal['reasoning'], Literal['agent']]):
    thought: str


Message = UserMessage | AgentTextMessage | AgentReasoningMessage

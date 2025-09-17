from abc import ABC
from abc import abstractmethod
import logging
from typing import Any
from typing import Generic
from typing import Literal
from typing import TypeVar

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import PrivateAttr

from tinygent.runtime.global_registry import GlobalRegistry

logger = logging.getLogger(__name__)

TinyMessageType = TypeVar(
    'TinyMessageType',
    Literal['chat'],
    Literal['tool'],
    Literal['human'],
    Literal['plan'],
)


class BaseMessage(ABC, BaseModel, Generic[TinyMessageType]):
    type: TinyMessageType

    metadata: dict = {}

    model_config = ConfigDict(frozen=True, extra='forbid')

    @property
    @abstractmethod
    def tiny_str(self) -> str:
        raise NotImplementedError('Subclasses must implement this method.')


class TinyPlanMessage(BaseMessage[Literal['plan']]):
    type: Literal['plan'] = 'plan'

    content: str

    metadata: dict = {}

    @property
    def tiny_str(self) -> str:
        return f'AI Plan: {self.content}'


class TinyChatMessage(BaseMessage[Literal['chat']]):
    type: Literal['chat'] = 'chat'

    content: str

    metadata: dict = {}

    @property
    def tiny_str(self) -> str:
        return f'AI: {self.content}'


class TinyToolCall(BaseMessage[Literal['tool']]):
    type: Literal['tool'] = 'tool'

    tool_name: str

    arguments: dict

    call_id: str | None = None

    _result: Any | None = PrivateAttr(default=None)

    metadata: dict = {}

    @property
    def result(self) -> Any | None:
        return self._result

    @property
    def tiny_str(self) -> str:
        result_str = (
            f' -> Result: {self.result}' if self.result is not None else 'No result'
        )

        return (
            '[EXECUTED] - ' if self.metadata.get('executed') else '[NOT EXECUTED] - '
        ) + f'Tool Call: {self.tool_name}({self.arguments}){result_str}'

    def call(self) -> None:
        tool = GlobalRegistry.get_registry().get_tool(self.tool_name)
        result = tool(**self.arguments)
        self.metadata['executed'] = True
        self._result = result


class TinyHumanMessage(BaseMessage[Literal['human']]):
    type: Literal['human'] = 'human'

    content: str

    metadata: dict = {}

    @property
    def tiny_str(self) -> str:
        return f'Human: {self.content}'


TinyAIMessage = TinyPlanMessage | TinyChatMessage | TinyToolCall

AllTinyMessages = TinyAIMessage | TinyHumanMessage

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
    Literal['reasoning'],
)


class BaseMessage(ABC, BaseModel, Generic[TinyMessageType]):
    """Abstract base class for all message types."""

    type: TinyMessageType
    """The type of the message."""

    metadata: dict = {}
    """Metadata associated with the message."""

    model_config = ConfigDict(frozen=True, extra='forbid')
    """Pydantic model configuration."""

    @property
    @abstractmethod
    def tiny_str(self) -> str:
        """A concise string representation of the message."""
        raise NotImplementedError('Subclasses must implement this method.')


class TinyPlanMessage(BaseMessage[Literal['plan']]):
    """Message representing the AI's plan."""

    type: Literal['plan'] = 'plan'
    """The type of the message."""

    content: str
    """The content of the plan message."""

    @property
    def tiny_str(self) -> str:
        return f'AI Plan: {self.content}'


class TinyReasoningMessage(BaseMessage[Literal['reasoning']]):
    """Message representing the AI's reasoning."""

    type: Literal['reasoning'] = 'reasoning'
    """The type of the message."""

    content: str
    """The content of the reasoning message."""

    @property
    def tiny_str(self) -> str:
        return f'AI Reasoning: {self.content}'


class TinyChatMessage(BaseMessage[Literal['chat']]):
    """Message representing a chat from the AI."""

    type: Literal['chat'] = 'chat'
    """The type of the message."""

    content: str
    """The content of the chat message."""

    @property
    def tiny_str(self) -> str:
        return f'AI: {self.content}'


class TinyToolCall(BaseMessage[Literal['tool']]):
    """Message representing a tool call from the AI."""

    type: Literal['tool'] = 'tool'
    """The type of the message."""

    tool_name: str
    """The name of the tool being called."""

    arguments: dict
    """The arguments for the tool call."""

    call_id: str | None = None
    """An optional identifier for the tool call."""

    _result: Any | None = PrivateAttr(default=None)
    """The result of the tool call, initially None."""

    @property
    def result(self) -> Any | None:
        """The result of the tool call."""
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
    """Message representing input from a human."""

    type: Literal['human'] = 'human'
    """The type of the message."""

    content: str
    """The content of the human message."""

    @property
    def tiny_str(self) -> str:
        return f'Human: {self.content}'


TinyAIMessage = TinyPlanMessage | TinyReasoningMessage | TinyChatMessage | TinyToolCall

AllTinyMessages = TinyAIMessage | TinyHumanMessage

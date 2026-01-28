from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import logging
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any
from typing import Generic
from typing import Literal
from typing import TypeVar

from pydantic import ConfigDict
from pydantic import Field
from pydantic import PrivateAttr

from tinygent.core.types.base import TinyModel

if TYPE_CHECKING:
    from tinygent.core.datamodels.tool import AbstractTool

logger = logging.getLogger(__name__)

TinyMessageType = TypeVar(
    'TinyMessageType',
    Literal['system'],
    Literal['squad_member'],
    Literal['chat'],
    Literal['tool'],
    Literal['tool_result'],
    Literal['human'],
    Literal['plan'],
    Literal['reasoning'],
    Literal['summary'],
)


class BaseMessage(ABC, TinyModel, Generic[TinyMessageType]):
    """Abstract base class for all message types."""

    type: TinyMessageType
    """The type of the message."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Metadata associated with the message."""

    model_config = ConfigDict(extra='forbid')
    """Pydantic model configuration."""

    @property
    @abstractmethod
    def tiny_str(self) -> str:
        """A concise string representation of the message."""
        raise NotImplementedError('Subclasses must implement this method.')


class TinySystemMessage(BaseMessage[Literal['system']]):
    """Message representing system-level instructions."""

    type: Literal['system'] = 'system'
    """The type of the message."""

    content: str
    """The content of the system message."""

    @property
    def tiny_str(self) -> str:
        return f'SYSTEM: {self.content}'


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


class TinySummaryMessage(BaseMessage[Literal['summary']]):
    """Message representing the AI's summary."""

    type: Literal['summary'] = 'summary'
    """The type of the message."""

    content: str
    """The content of the summary message."""

    @property
    def tiny_str(self) -> str:
        return f'AI Summary: {self.content}'


class TinyChatMessage(BaseMessage[Literal['chat']]):
    """Message representing a chat from the AI."""

    type: Literal['chat'] = 'chat'
    """The type of the message."""

    content: str
    """The content of the chat message."""

    @property
    def tiny_str(self) -> str:
        return f'AI: {self.content}'


class TinyChatMessageChunk(BaseMessage[Literal['chat']]):
    """Message representing a chunk of a chat from the AI."""

    type: Literal['chat'] = 'chat'
    """The type of the message."""

    content: str
    """The content chunk of the chat message."""

    @property
    def tiny_str(self) -> str:
        return f'AI Chunk: {self.content}'


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

    @result.setter
    def result(self, value: Any) -> None:
        self._result = value

    @property
    def tiny_str(self) -> str:
        result_str = (
            f' -> Result: {self.result}' if self.result is not None else 'No result'
        )

        return (
            '[EXECUTED] - ' if self.metadata.get('executed') else '[NOT EXECUTED] - '
        ) + f'Tool Call: {self.tool_name}({self.arguments}){result_str}'


class TinySquadMemberMessage(BaseMessage[Literal['squad_member']]):
    """Message representing input from a squad member agent."""

    type: Literal['squad_member'] = 'squad_member'
    """The type of the message."""

    member_name: str
    """The name of the squad member."""

    task: str
    """The task assigned to the squad member."""

    result: str
    """The result produced by the squad member."""

    @property
    def tiny_str(self) -> str:
        return (
            f'Squad Member {self.member_name} - Task: {self.task}, Result: {self.result}'
        )


class TinyToolCallChunk(BaseMessage[Literal['tool']]):
    """Message representing a chunk of a tool call from the AI."""

    type: Literal['tool'] = 'tool'
    """The type of the message."""

    tool_name: str | None = None
    """The name of the tool being called."""

    arguments: str | None = None
    """The arguments chunk for the tool call."""

    call_id: str | None = None
    """An optional identifier for the tool call."""

    index: int
    """The index of the tool call in the message stream."""

    @property
    def tiny_str(self) -> str:
        return f'Tool Call Chunk: {self.tool_name or "?"}({self.arguments or ""})'


class TinyToolResult(BaseMessage[Literal['tool_result']]):
    """Message representing the result of a tool call."""

    type: Literal['tool_result'] = 'tool_result'
    """The type of the message."""

    call_id: str
    """The identifier of the tool call."""

    content: str
    """The content of the tool result message."""

    _raw: AbstractTool = PrivateAttr()
    """The raw tool that produced the result."""

    @property
    def tiny_str(self) -> str:
        return f'Tool Result - {self.call_id}: {self.content}'

    @property
    def raw(self) -> AbstractTool:
        """The raw tool that produced the result."""
        return self._raw

    @raw.setter
    def raw(self, tool: AbstractTool) -> None:
        """Set the raw tool that produced the result."""
        self._raw = tool


class TinyHumanMessage(BaseMessage[Literal['human']]):
    """Message representing input from a human."""

    type: Literal['human'] = 'human'
    """The type of the message."""

    content: str
    """The content of the human message."""

    @property
    def tiny_str(self) -> str:
        return f'Human: {self.content}'


TinyAIMessage = (
    TinyPlanMessage
    | TinyReasoningMessage
    | TinyChatMessage
    | TinyToolCall
    | TinySquadMemberMessage
    | TinySummaryMessage
)

AllTinyMessages = Annotated[
    (
        TinyPlanMessage
        | TinyReasoningMessage
        | TinyChatMessage
        | TinyToolCall
        | TinySquadMemberMessage
        | TinyHumanMessage
        | TinySystemMessage
        | TinyToolResult
        | TinySummaryMessage
    ),
    Field(discriminator='type'),
]

TinyAIMessageChunk = TinyChatMessageChunk | TinyToolCallChunk

AllTinyMessageChunks = TinyAIMessageChunk

from typing import Literal
from tinygent.datamodels.messages import TinyChatMessageChunk, TinyToolCall, TinyToolCallChunk
from tinygent.types.base import TinyModel


class TinyLLMResultChunk(TinyModel):
    """A chunk of an LLM result, consisting of a single message."""

    type: Literal['message', 'tool_call', 'end']

    message: TinyChatMessageChunk | None = None
    tool_call: TinyToolCallChunk | None = None
    full_tool_call: TinyToolCall | None = None

    metadata: dict | None = None

    @property
    def is_end(self) -> bool:
        """Check if this chunk indicates the end of the stream."""
        return self.type == 'end'

    @property
    def is_message(self) -> bool:
        """Check if this chunk is a message."""
        return self.type == 'message'

    @property
    def is_tool_call(self) -> bool:
        """Check if this chunk is a tool call."""
        return self.type == 'tool_call'

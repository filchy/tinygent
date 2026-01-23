from typing import Literal

from tinygent.core.datamodels.messages import TinyChatMessageChunk
from tinygent.core.datamodels.messages import TinyToolCall
from tinygent.core.datamodels.messages import TinyToolCallChunk
from tinygent.core.types.base import TinyModel


class TinyLLMResultChunk(TinyModel):
    """A chunk of an LLM result, consisting of a single message."""

    type: Literal['message', 'tool_call', 'end', 'none']

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

    def to_string(self) -> str:
        """Convert the chunk to a string representation."""
        parts: list[str] = [f'type={self.type}']

        if self.message:
            parts.append(f'message={self.message.tiny_str}')
        if self.tool_call:
            parts.append(f'tool_call={self.tool_call.tiny_str}')
        if self.full_tool_call:
            parts.append(f'full_tool_call={self.full_tool_call.tiny_str}')

        return f'TinyLLMResultChunk({", ".join(parts)})'

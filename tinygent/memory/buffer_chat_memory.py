from io import StringIO
from typing import Literal

from pydantic import Field

from tinygent.core.datamodels.memory import AbstractMemoryConfig
from tinygent.memory import BaseChatMemory


class BufferChatMemoryConfig(AbstractMemoryConfig['BufferChatMemory']):
    type: Literal['buffer'] = Field(default='buffer', frozen=True)

    def build(self) -> 'BufferChatMemory':
        return BufferChatMemory()


class BufferChatMemory(BaseChatMemory):
    """Simple buffer memory that stores full conversation history.

    Maintains the complete chat history without any truncation or summarization.
    All messages are stored in memory and returned when loading variables.

    This is the simplest memory type, suitable for:
    - Short conversations that fit within context limits
    - Scenarios where full history is required
    - Development and testing with small message counts

    Note: This memory type can exceed context limits for long conversations.
    Consider BufferWindowChatMemory or BufferSummaryChatMemory for longer
    interactions.
    """

    def __init__(self) -> None:
        super().__init__()

        self._memory_key: str = 'full_chat_history'

    @property
    def memory_keys(self) -> list[str]:
        return [self._memory_key]

    def load_variables(self) -> dict[str, str]:
        return {
            self._memory_key: str([msg.tiny_str for msg in self._chat_history.messages])
        }

    def __str__(self) -> str:
        base = super().__str__()

        buff = StringIO()

        buff.write(base)
        buff.write('\ttype: Buffer Chat Memory\n')

        return buff.getvalue()

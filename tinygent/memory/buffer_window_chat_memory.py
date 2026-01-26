from io import StringIO
from typing import Literal

from pydantic import Field

from tinygent.core.datamodels.memory import AbstractMemoryConfig
from tinygent.core.datamodels.messages import AllTinyMessages
from tinygent.memory import BaseChatMemory


class BufferWindowChatMemoryConfig(AbstractMemoryConfig['BufferWindowChatMemory']):
    type: Literal['buffer_window'] = Field(default='buffer_window', frozen=True)

    k: int = Field(default=5)

    def build(self) -> 'BufferWindowChatMemory':
        return BufferWindowChatMemory(k=self.k)


class BufferWindowChatMemory(BaseChatMemory):
    """Sliding window memory that keeps only the last K messages.

    Maintains a fixed-size window of recent conversation history by keeping
    only the last K messages. This prevents memory from growing unbounded
    while preserving recent context.

    The window slides forward as new messages arrive - when the limit is reached,
    the oldest message is dropped to make room for new ones.

    Suitable for:
    - Long-running conversations with context limits
    - Scenarios where only recent context is relevant
    - Preventing token limits from being exceeded

    Args:
        k: Number of messages to retain (default: 5)
    """

    def __init__(self, k: int = 5) -> None:
        super().__init__()

        self.k = k

    @property
    def _memory_key(self) -> str:
        return f'last_{self.k}_messages_window'

    @property
    def chat_buffer_window(self) -> list[AllTinyMessages]:
        return self._chat_history.messages[-self.k :] if self.k > 0 else []

    @property
    def memory_keys(self) -> list[str]:
        return [self._memory_key]

    def load_variables(self) -> dict[str, str]:
        return {self._memory_key: str([msg.tiny_str for msg in self.chat_buffer_window])}

    def __str__(self) -> str:
        base = super().__str__()

        buff = StringIO()

        buff.write(base)
        buff.write('\ttype: Window Buffer Chat Memory:\n')
        buff.write(f'\tWindow size (k): {self.k}\n')

        return buff.getvalue()

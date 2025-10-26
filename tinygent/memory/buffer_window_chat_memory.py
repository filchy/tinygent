from io import StringIO
from typing import Literal

from tinygent.datamodels.memory import AbstractMemoryConfig
from tinygent.datamodels.messages import AllTinyMessages
from tinygent.memory import BaseChatMemory


class BufferWindowChatMemoryConfig(AbstractMemoryConfig['BufferWindowChatMemory']):
    type: Literal['buffer_window'] = 'buffer_window'

    k: int = 5

    def build(self) -> 'BufferWindowChatMemory':
        return BufferWindowChatMemory(k=self.k)


class BufferWindowChatMemory(BaseChatMemory):
    k: int = 5
    _memory_key: str = f'last_{k}_messages_window'

    @property
    def chat_buffer_window(self) -> list[AllTinyMessages]:
        return self._chat_history.messages[-self.k :] if self.k > 0 else []

    @property
    def memory_keys(self) -> list[str]:
        return [self._memory_key]

    def load_variables(self) -> dict[str, str]:
        return {self._memory_key: str(self.chat_buffer_window)}

    def __str__(self) -> str:
        base = super().__str__()

        buff = StringIO()

        buff.write(base)
        buff.write('\ttype: Window Buffer Chat Memory:\n')
        buff.write(f'\tWindow size (k): {self.k}\n')

        return buff.getvalue()

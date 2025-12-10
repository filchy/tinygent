from io import StringIO
from typing import Literal

from tinygent.datamodels.memory import AbstractMemoryConfig
from tinygent.memory import BaseChatMemory


class BufferChatMemoryConfig(AbstractMemoryConfig['BufferChatMemory']):
    type: Literal['buffer'] = 'buffer'

    def build(self) -> 'BufferChatMemory':
        return BufferChatMemory()


class BufferChatMemory(BaseChatMemory):
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

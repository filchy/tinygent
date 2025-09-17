from tinygent.datamodels.messages import AllTinyMessages
from tinygent.memory.base_chat_memory import BaseChatMemory


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

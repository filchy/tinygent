from tinygent.memory.base_chat_memory import BaseChatMemory


class BufferChatMemory(BaseChatMemory):
    _memory_key: str = 'full_chat_history'

    @property
    def memory_keys(self) -> list[str]:
        return [self._memory_key]

    def load_variables(self) -> dict[str, str]:
        return {self._memory_key: str(self._chat_history)}

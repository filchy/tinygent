from pydantic import PrivateAttr

from tinygent.core.chat_history import BaseChatHistory
from tinygent.datamodels.llm_io import TinyLLMInput
from tinygent.datamodels.llm_io import TinyLLMResult
from tinygent.datamodels.memory import AbstractMemory


class BaseChatMemory(AbstractMemory):
    _chat_history: BaseChatHistory = PrivateAttr(default_factory=BaseChatHistory)

    @property
    def memory_keys(self) -> list[str]:
        return ['chat_history']

    def load_variables(self) -> dict[str, str]:
        return {'chat_history': str(self._chat_history)}

    def save_context(self, input: TinyLLMInput, output: TinyLLMResult) -> None:
        all_messages = [input.to_tiny_message()] + list(output.tiny_iter())

        self._chat_history.add_messages(all_messages)

    def clear(self) -> None:
        self._chat_history.clear()

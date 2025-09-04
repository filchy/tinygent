from pydantic import PrivateAttr

from tinygent.core.chat_history import BaseChatHistory
from tinygent.datamodels.memory import AbstractMemory
from tinygent.datamodels.messages import AllTinyMessages


class BaseChatMemory(AbstractMemory):
    _chat_history: BaseChatHistory = PrivateAttr(default_factory=BaseChatHistory)

    @property
    def memory_keys(self) -> list[str]:
        return ['chat_history']

    def load_variables(self) -> dict[str, str]:
        return {'chat_history': str(self._chat_history)}

    def save_context(self, message: AllTinyMessages) -> None:
        self._chat_history.add_message(message)

    def clear(self) -> None:
        self._chat_history.clear()

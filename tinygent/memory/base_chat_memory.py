from __future__ import annotations

from abc import ABC
import typing

from pydantic import PrivateAttr

from tinygent.core.chat_history import BaseChatHistory
from tinygent.datamodels.memory import AbstractMemory

if typing.TYPE_CHECKING:
    from tinygent.datamodels.messages import AllTinyMessages


class BaseChatMemory(AbstractMemory, ABC):
    _chat_history: BaseChatHistory = PrivateAttr(default_factory=BaseChatHistory)

    @property
    def chat_messages(self) -> list[AllTinyMessages]:
        return self._chat_history.messages

    def save_context(self, message: AllTinyMessages) -> None:
        self._chat_history.add_message(message)

    def clear(self) -> None:
        self._chat_history.clear()

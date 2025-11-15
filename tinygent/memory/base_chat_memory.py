from __future__ import annotations

from abc import ABC
from io import StringIO
import typing

from pydantic import PrivateAttr

from tinygent.core.chat_history import BaseChatHistory
from tinygent.datamodels.memory import AbstractMemory
from tinygent.utils.pydantic_utils import tiny_deep_copy

if typing.TYPE_CHECKING:
    from tinygent.datamodels.messages import AllTinyMessages


class BaseChatMemory(AbstractMemory, ABC):
    _chat_history: BaseChatHistory = PrivateAttr(default_factory=BaseChatHistory)

    @property
    def copy_chat_messages(self) -> list[AllTinyMessages]:
        return [tiny_deep_copy(msg) for msg in self._chat_history.messages]

    def save_context(self, message: AllTinyMessages) -> None:
        self._chat_history.add_message(message)

    def clear(self) -> None:
        self._chat_history.clear()

    def __str__(self) -> str:
        buff = StringIO()

        buff.write('Chat Memory:\n')
        buff.write(f'\tNumber of messages stored: {len(self._chat_history.messages)}\n')

        return buff.getvalue()

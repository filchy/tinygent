from __future__ import annotations

from abc import ABC
from io import StringIO
import typing

from tinygent.core.chat_history import BaseChatHistory
from tinygent.core.datamodels.memory import AbstractMemory
from tinygent.utils.pydantic_utils import tiny_deep_copy

if typing.TYPE_CHECKING:
    from tinygent.core.datamodels.messages import AllTinyMessages


class BaseChatMemory(AbstractMemory, ABC):
    def __init__(self) -> None:
        self._chat_history: BaseChatHistory = BaseChatHistory()

    def copy_chat_messages(self) -> list[AllTinyMessages]:
        return [tiny_deep_copy(msg) for msg in self._chat_history.messages]  # type: ignore

    def save_context(self, message: AllTinyMessages) -> None:
        self._chat_history.add_message(message)

    def save_multiple_context(self, messages: list[AllTinyMessages]) -> None:
        for msg in messages:
            self.save_context(msg)

    def clear(self) -> None:
        self._chat_history.clear()

    def __str__(self) -> str:
        buff = StringIO()

        buff.write('Chat Memory:\n')
        buff.write(f'\tNumber of messages stored: {len(self._chat_history.messages)}\n')

        return buff.getvalue()

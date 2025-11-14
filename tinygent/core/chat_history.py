from typing import Callable
from typing import Sequence

from pydantic import PrivateAttr

from tinygent.datamodels.messages import AllTinyMessages
from tinygent.datamodels.messages import TinyAIMessage
from tinygent.datamodels.messages import TinyChatMessage
from tinygent.datamodels.messages import TinyHumanMessage
from tinygent.types import TinyModel


class BaseChatHistory(TinyModel):
    _messages: list[AllTinyMessages] = PrivateAttr(default_factory=list)
    _filters: dict[str, Callable[[AllTinyMessages], bool]] = PrivateAttr(
        default_factory=dict
    )

    @property
    def messages(self) -> list[AllTinyMessages]:
        if not self._filters:
            return self._messages
        return [m for m in self._messages if all(f(m) for f in self._filters.values())]

    @messages.setter
    def messages(self, value: list[AllTinyMessages]) -> None:
        raise ValueError(
            "Direct assignment to 'messages' is not allowed. Use 'add_message' or 'add_messages' methods."
        )

    def add_message(self, message: AllTinyMessages) -> None:
        self.messages.append(message)

    def add_messages(self, messages: Sequence[AllTinyMessages]) -> None:
        self.messages.extend(messages)

    def add_ai_message(self, message: TinyAIMessage | str) -> None:
        if isinstance(message, str):
            message = TinyChatMessage(content=message)

        self.messages.append(message)

    def add_human_message(self, message: str | TinyHumanMessage) -> None:
        if isinstance(message, str):
            message = TinyHumanMessage(content=message)

        self.messages.append(message)

    def clear(self) -> None:
        self.messages.clear()

    def add_filter(self, name: str, func: Callable[[AllTinyMessages], bool]) -> None:
        if name in self._filters:
            raise ValueError(f"Filter with name '{name}' already exists.")
        self._filters[name] = func

    def remove_filter(self, name: str) -> None:
        if name not in self._filters:
            raise ValueError(f"Filter with name '{name}' does not exist.")
        self._filters.pop(name)

    def list_filters(self) -> list[str]:
        return list(self._filters.keys())

    def __str__(self) -> str:
        parts = []

        for message in self.messages:
            try:
                tiny_message = message.tiny_str
            except NotImplementedError:
                tiny_message = 'Unknown message'

            parts.append(tiny_message)

        return '\n'.join(parts)

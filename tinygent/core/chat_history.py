import logging
from typing import Callable
from typing import Sequence

from pydantic import PrivateAttr

from tinygent.core.datamodels.messages import AllTinyMessages
from tinygent.core.datamodels.messages import TinyAIMessage
from tinygent.core.datamodels.messages import TinyChatMessage
from tinygent.core.datamodels.messages import TinyHumanMessage
from tinygent.core.types.base import TinyModel

logger = logging.getLogger(__name__)


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
        logger.debug('Adding message to chat history: %s', message)

        self.messages.append(message)

    def add_messages(self, messages: Sequence[AllTinyMessages]) -> None:
        logger.debug('Adding multiple messages to chat history: %s', messages)

        self.messages.extend(messages)

    def add_ai_message(self, message: TinyAIMessage | str) -> None:
        logger.debug('Adding AI message to chat history: %s', message)

        if isinstance(message, str):
            message = TinyChatMessage(content=message)

        self.messages.append(message)

    def add_human_message(self, message: str | TinyHumanMessage) -> None:
        logger.debug('Adding human message to chat history: %s', message)

        if isinstance(message, str):
            message = TinyHumanMessage(content=message)

        self.messages.append(message)

    def clear(self) -> None:
        logger.debug('Clearing chat history')
        self.messages.clear()

    def add_filter(self, name: str, func: Callable[[AllTinyMessages], bool]) -> None:
        logger.debug('Adding filter to chat history: %s', name)

        if name in self._filters:
            raise ValueError(f"Filter with name '{name}' already exists.")
        self._filters[name] = func

    def remove_filter(self, name: str) -> None:
        logger.debug('Removing filter from chat history: %s', name)

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

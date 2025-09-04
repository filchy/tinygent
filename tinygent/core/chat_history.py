from typing import Sequence

from pydantic import BaseModel
from pydantic import Field

from tinygent.datamodels.messages import AllTinyMessages
from tinygent.datamodels.messages import TinyAIMessage
from tinygent.datamodels.messages import TinyChatMessage
from tinygent.datamodels.messages import TinyHumanMessage


class BaseChatHistory(BaseModel):
    messages: list[AllTinyMessages] = Field(default_factory=list)

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

    def __str__(self) -> str:
        parts = []

        for message in self.messages:
            try:
                tiny_message = message.tiny_str
            except NotImplementedError:
                tiny_message = 'Unknown message'

            parts.append(tiny_message)

        return '\n'.join(parts)

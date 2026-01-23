from tinygent.core.datamodels.messages import AllTinyMessages
from tinygent.core.types.base import TinyModel


class TinyLLMInput(TinyModel):
    """Input to an LLM, consisting of a list of messages."""

    messages: list[AllTinyMessages] = []

    def add_before_last(self, message: AllTinyMessages) -> None:
        """Add a message before the last message in the list."""
        if not self.messages:
            self.messages.append(message)
        else:
            self.messages.insert(-1, message)

    def add_at_beginning(self, message: AllTinyMessages) -> None:
        """Add a message at the beginning of the list."""
        self.messages.insert(0, message)

    def add_at_end(self, message: AllTinyMessages) -> None:
        """Add a message at the end of the list."""
        self.messages.append(message)

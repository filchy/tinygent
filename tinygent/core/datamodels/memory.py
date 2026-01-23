from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Generic
from typing import TypeVar

from tinygent.core.datamodels.messages import AllTinyMessages
from tinygent.core.runtime.executors import run_sync_in_executor
from tinygent.core.types.builder import TinyModelBuildable

T = TypeVar('T', bound='AbstractMemory')


class AbstractMemoryConfig(TinyModelBuildable[T], Generic[T]):
    """Abstract base class for memory configurations."""

    def build(self) -> T:
        """Build the memory instance from the configuration."""
        raise NotImplementedError('Subclasses must implement this method.')


class AbstractMemory(ABC):
    """Abstract base class for memory modules."""

    @abstractmethod
    def copy_chat_messages(self) -> list[AllTinyMessages]:
        """Return a copy of the chat messages stored in memory."""
        raise NotImplementedError('Subclasses must implement this method.')

    @property
    @abstractmethod
    def memory_keys(self) -> list[str]:
        """List of keys used in the memory."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def load_variables(self) -> dict[str, Any]:
        """Load variables from memory."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def save_context(self, message: AllTinyMessages) -> None:
        """Save the context of a conversation to memory."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def save_multiple_context(self, messages: list[AllTinyMessages]) -> None:
        """Save multiple messages to memory."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def clear(self) -> None:
        """Clear the memory."""
        raise NotImplementedError('Subclasses must implement this method.')

    async def aload_variables(self) -> dict[str, str]:
        """Asynchronously load variables from memory."""
        return await run_sync_in_executor(self.load_variables)

    async def asave_context(self, message: AllTinyMessages) -> None:
        """Asynchronously save the context of a conversation to memory."""
        return await run_sync_in_executor(self.save_context, message)

    async def asave_multiple_context(self, messages: list[AllTinyMessages]) -> None:
        """Asynchronously save multiple messages to memory."""
        return await run_sync_in_executor(self.save_multiple_context, messages)

    async def aclear(self) -> None:
        """Asynchronously clear the memory."""
        return await run_sync_in_executor(self.clear)

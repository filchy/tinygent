from abc import ABC
from abc import abstractmethod

from tinygent.datamodels.messages import AllTinyMessages
from tinygent.runtime.executors import run_sync_in_executor
from tinygent.types.base import TinyModel


class AbstractMemory(TinyModel, ABC):
    """Abstract base class for memory modules."""

    @property
    @abstractmethod
    def memory_keys(self) -> list[str]:
        """List of keys used in the memory."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def load_variables(self) -> dict[str, str]:
        """Load variables from memory."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def save_context(self, message: AllTinyMessages) -> None:
        """Save the context of a conversation to memory."""
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

    async def aclear(self) -> None:
        """Asynchronously clear the memory."""
        return await run_sync_in_executor(self.clear)

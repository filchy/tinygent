from abc import ABC
from abc import abstractmethod

from pydantic import BaseModel

from tinygent.datamodels.messages import AllTinyMessages
from tinygent.runtime.executors import run_sync_in_executor


class AbstractMemory(BaseModel, ABC):
    """Abstract base class for memory modules."""

    @property
    @abstractmethod
    def memory_keys(self) -> list[str]: ...

    """List of keys used in the memory."""

    @abstractmethod
    def load_variables(self) -> dict[str, str]: ...

    """Load variables from memory."""

    @abstractmethod
    def save_context(self, message: AllTinyMessages) -> None: ...

    """Save the context of a conversation to memory."""

    @abstractmethod
    def clear(self) -> None: ...

    """Clear the memory."""

    async def aload_variables(self) -> dict[str, str]:
        """Asynchronously load variables from memory."""
        return await run_sync_in_executor(self.load_variables)

    async def asave_context(self, message: AllTinyMessages) -> None:
        """Asynchronously save the context of a conversation to memory."""
        return await run_sync_in_executor(self.save_context, message)

    async def aclear(self) -> None:
        """Asynchronously clear the memory."""
        return await run_sync_in_executor(self.clear)

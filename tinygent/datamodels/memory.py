from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Generic
from typing import TypeVar

from tinygent.datamodels.messages import AllTinyMessages
from tinygent.runtime.executors import run_sync_in_executor
from tinygent.types.base import TinyModel
from tinygent.types.builder import TinyModelBuildable

T = TypeVar('T', bound='AbstractMemory')


class AbstractMemoryConfig(TinyModelBuildable[T], Generic[T]):
    """Abstract base class for memory configurations."""

    type: Any  # used as discriminator

    _discriminator_field: str = 'type'

    @classmethod
    def get_discriminator_field(cls) -> str:
        """Get the name of the discriminator field."""
        return cls._discriminator_field

    def build(self) -> T:
        """Build the memory instance from the configuration."""
        raise NotImplementedError('Subclasses must implement this method.')


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

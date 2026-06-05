from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Generic
from typing import TypeVar

from tinygent.core.types.builder import TinyModelBuildable

T = TypeVar('T', bound='AbstractCheckpointer')


class AbstractCheckpointerConfig(TinyModelBuildable[T], Generic[T]):
    """Abstract base class for checkpoint configuration."""

    def build(self) -> T:
        """Build the checkpointer instance from the configuration."""
        raise NotImplementedError('Subclasses must implement this method.')


class AbstractCheckpointer(ABC):
    """Abstract base class for checkpoint middleware."""

    @abstractmethod
    def save(self, checkpoint_id: str) -> None:
        """Save current checkpoint state."""
        pass

    @abstractmethod
    def load(self, checkpoint_id: str) -> None:
        """Load saved checkpoint state."""
        pass

    @abstractmethod
    def delete(self, checkpoint_id: str) -> None:
        """Delete desired checkpoint."""
        pass

    @abstractmethod
    def set_data(self, data: dict[str, Any]) -> None:
        """Set checkpoint data manyally."""
        pass

    @abstractmethod
    def setdefault(self, key: str, value: Any) -> Any:
        """Initialize a key only if missing, then return the stored value.

        Non-destructive: keys already present (e.g. from a loaded checkpoint)
        are left untouched. This is the building block for "set sometimes,
        use sometimes" state initialization.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Drop all checkpoint data (explicit fresh-start)."""
        pass

    @abstractmethod
    def __getitem__(self, key: str, default: Any = None) -> Any:
        """Get checkpoint data by key."""
        pass

    @abstractmethod
    def __setitem__(self, key: str, value: Any) -> None:
        """Set checkpoint data by key."""
        pass

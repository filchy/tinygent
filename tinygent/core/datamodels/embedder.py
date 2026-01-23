from abc import ABC
from abc import abstractmethod
from typing import Generic
from typing import TypeVar

from pydantic import SecretStr

from tinygent.core.types.builder import TinyModelBuildable

T = TypeVar('T', bound='AbstractEmbedder')


class AbstractEmbedderConfig(TinyModelBuildable[T], Generic[T]):
    """Abstract base class for Embedder configuration."""

    model: str

    api_key: SecretStr | None

    def build(self) -> T:
        """Build the Embedder instance from the configuration."""
        raise NotImplementedError('Subclasses must implement this method.')


class AbstractEmbedder(ABC):
    """Abstract base class for Embedders."""

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the Embedder with the given configuration."""
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Returns current model name."""
        raise NotImplementedError('Subclasses must implement this method.')

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Get dimension of the embeddings for the current model."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def embed(self, query: str) -> list[float]:
        """Create embedding for single input query."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def embed_batch(self, queries: list[str]) -> list[list[float]]:
        """Create embedding for batch of queries."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    async def aembed(self, query: str) -> list[float]:
        """Create embedding for single input query."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    async def aembed_batch(self, queries: list[str]) -> list[list[float]]:
        """Create embedding for batch of queries."""
        raise NotImplementedError('Subclasses must implement this method.')

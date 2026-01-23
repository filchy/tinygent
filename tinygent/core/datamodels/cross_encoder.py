from abc import ABC
from abc import abstractmethod
from typing import Generic
from typing import Iterable
from typing import TypeVar

from tinygent.core.types.builder import TinyModelBuildable

T = TypeVar('T', bound='AbstractCrossEncoder')


class AbstractCrossEncoderConfig(TinyModelBuildable[T], Generic[T]):
    """Abstract base class for Cross-encoder configuration."""

    def build(self) -> T:
        """Build the Cross-encoder instance from the configuration."""
        raise NotImplementedError('Subclasses must implement this method.')


class AbstractCrossEncoder(ABC):
    """Abstract base class for Cross-encoder."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the Cross-encoder with the given configuration."""
        pass

    @abstractmethod
    async def rank(
        self, query: str, texts: Iterable[str]
    ) -> list[tuple[tuple[str, str], float]]:
        """Rank a list of texts agains given query."""
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    async def predict(
        self, pairs: Iterable[tuple[str, str]]
    ) -> list[tuple[tuple[str, str], float]]:
        """Predict scores for a pair of senteces."""
        raise NotImplementedError('Subclasses must implement this method.')

from abc import ABC
from abc import abstractmethod
from typing import Any

from typing_extensions import LiteralString

from tiny_graph.types.provider import GraphProvider


class BaseDriver(ABC):
    """Abstract base class for graph db drivers."""

    provider: GraphProvider

    @abstractmethod
    async def execute_query(self, query: str | LiteralString, **kwargs: Any) -> Any:
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError('Subclasses must implement this method.')

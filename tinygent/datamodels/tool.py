from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import typing
from typing import Any
from typing import Callable
from typing import TypeVar

if typing.TYPE_CHECKING:
    from tinygent.datamodels.tool_info import ToolInfo

T = TypeVar('T', bound='AbstractTool')


class AbstractTool(ABC):
    """Abstract base class for tools."""

    @abstractmethod
    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Initialize the tool with the given function and arguments."""
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the tool with the given arguments."""
        pass

    @property
    @abstractmethod
    def info(self) -> ToolInfo:
        """Get the information about the tool."""
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear the tool's cache."""
        pass

    @abstractmethod
    def cache_info(self) -> Any:
        """Get information about the tool's cache."""
        pass

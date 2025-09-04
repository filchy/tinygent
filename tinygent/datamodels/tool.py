from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Callable

from tinygent.datamodels.tool_info import ToolInfo


class AbstractTool(ABC):
    @abstractmethod
    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None: ...

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    @property
    @abstractmethod
    def info(self) -> ToolInfo: ...

    @abstractmethod
    def clear_cache(self) -> None: ...

    @abstractmethod
    def cache_info(self) -> Any: ...

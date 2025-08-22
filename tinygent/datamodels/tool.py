from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Callable


class AbstractTool(ABC):

    @abstractmethod
    def __init__(self, fn: Callable[..., Any]) -> None:
        ...

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...

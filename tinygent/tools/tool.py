import dataclasses
import functools
from typing import Any
from typing import Callable
from typing import cast


@dataclasses.dataclass
class ToolInfo:

    arg_count: int

    @classmethod
    def from_callable(cls, fn: Callable[..., Any]) -> 'ToolInfo':

        return cls(
            arg_count=fn.__code__.co_argcount
        )


class Tool(ToolInfo):

    def __init__(self, fn: Callable[..., Any]) -> None:

        self._fn = fn
        self._info = ToolInfo.from_callable(fn)

    @property
    def info(self) -> ToolInfo:

        return self._info

    def __call__(self, *args: Any, **kwargs: Any) -> Any:

        return self._fn(*args, **kwargs)


def tool(fn: Callable[..., Any]) -> Tool:

    wrapped = Tool(fn)
    functools.update_wrapper(cast(Callable[..., Any], wrapped), fn)
    return wrapped

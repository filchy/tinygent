import sys
import dataclasses
import functools
import inspect

from pydantic import BaseModel
from pydantic import create_model
from typing import Any, get_origin
from typing import Callable
from typing import cast
from typing import TextIO

from pydantic_core import PydanticUndefined


@dataclasses.dataclass
class ToolInfo:

    name: str

    description: str

    arg_count: int

    is_coroutine: bool

    is_async_generator: bool

    input_schema: type[BaseModel] | None

    output_schema: type[BaseModel] | None

    @classmethod
    def from_callable(cls, fn: Callable[..., Any]) -> 'ToolInfo':

        name = fn.__name__
        description = inspect.getdoc(fn) or ''

        is_coroutin = inspect.iscoroutinefunction(fn)
        is_async_gen = inspect.isasyncgenfunction(fn)

        sig = inspect.signature(fn)

        arg_count = len(sig.parameters)

        if arg_count > 0:
            fields: dict[str, Any] = {}

            for param in sig.parameters.values():

                annotation = param.annotation
                if (
                    annotation is inspect.Parameter.empty
                    or (
                        not isinstance(annotation, type)
                        and get_origin(annotation) is None
                    )
                ):
                    annotation = Any

                default = param.default
                if default is inspect.Parameter.empty:
                    default = ...

                fields[param.name] = (annotation, default)

            input_schema = create_model(
                'ToolInputArgs',
                **fields
            )
        else:
            input_schema = None

        return cls(
            name=name,
            description=description,
            arg_count=arg_count,
            is_coroutine=is_coroutin,
            is_async_generator=is_async_gen,
            input_schema=input_schema,
            output_schema=None
        )

    def print_summary(self, stream: TextIO = sys.stdout):

        stream.write("Tool Summary:\n")
        stream.write("-" * 20 + "\n")

        stream.write(f"Name: {self.name}\n")
        stream.write(f"Description: {self.description}\n")
        stream.write(f"Argument Count: {self.arg_count}\n")
        stream.write(f"Is Coroutine: {self.is_coroutine}\n")
        stream.write(f"Is Async Generator: {self.is_async_generator}\n")
        stream.write("-" * 20 + "\n")


class Tool:

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

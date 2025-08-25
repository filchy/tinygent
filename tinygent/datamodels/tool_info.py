import dataclasses
import inspect
import sys

from typing import Any
from typing import get_origin
from typing import Callable
from typing import TextIO
from pydantic import BaseModel
from pydantic import create_model


@dataclasses.dataclass
class ToolInfo:

    name: str

    description: str

    arg_count: int

    is_coroutine: bool

    is_generator: bool

    is_async_generator: bool

    input_schema: type[BaseModel] | None

    output_schema: type[BaseModel] | None

    required_fields: list[str] = dataclasses.field(default_factory=list)

    @classmethod
    def from_callable(cls, fn: Callable[..., Any]) -> 'ToolInfo':

        name = fn.__name__
        description = inspect.getdoc(fn) or ''

        is_coroutin = inspect.iscoroutinefunction(fn)
        is_generator = inspect.isgeneratorfunction(fn)
        is_async_generator = inspect.isasyncgenfunction(fn)

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

            required_fields = [
                name for name, (_, default) in fields.items()
                if default is ...
            ]
        else:
            input_schema = None
            required_fields = []

        return_annotation = sig.return_annotation

        if (
            return_annotation is not inspect.Signature.empty
            and return_annotation is not type(None)  # noqa
            and not is_generator
            and not is_async_generator
        ):
            try:
                output_schema = create_model(
                    "ToolOutput",
                    __root__=(return_annotation, ...)
                )
            except Exception:
                output_schema = None
        else:
            output_schema = None

        return cls(
            name=name,
            description=description,
            arg_count=arg_count,
            is_coroutine=is_coroutin,
            is_generator=is_generator,
            is_async_generator=is_async_generator,
            input_schema=input_schema,
            output_schema=output_schema,
            required_fields=required_fields
        )

    def print_summary(self, stream: TextIO = sys.stdout):

        stream.write("Tool Summary:\n")
        stream.write("-" * 20 + "\n")

        stream.write(f"Name: {self.name}\n")
        stream.write(f"Description: {self.description}\n")
        stream.write(f"Argument Count: {self.arg_count}\n")
        stream.write(f"Is Coroutine: {self.is_coroutine}\n")
        stream.write(f"Is Generator: {self.is_generator}\n")
        stream.write(f"Is Async Generator: {self.is_async_generator}\n")
        stream.write("-" * 20 + "\n")


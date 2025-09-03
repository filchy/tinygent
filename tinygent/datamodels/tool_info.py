import dataclasses
import inspect
import sys
from typing import Callable
from typing import Generic
from typing import TextIO
from typing import TypeVar
from typing import cast

from pydantic import BaseModel

P = TypeVar('P', bound=BaseModel)
R = TypeVar('R')


@dataclasses.dataclass
class ToolInfo(Generic[P, R]):

    name: str

    description: str

    arg_count: int

    is_coroutine: bool

    is_generator: bool

    is_async_generator: bool

    input_schema: type[P] | None

    output_schema: type[BaseModel] | None

    required_fields: list[str] = dataclasses.field(default_factory=list)

    @classmethod
    def from_callable(cls, fn: Callable[[P], R]) -> 'ToolInfo[P, R]':

        name = fn.__name__
        description = inspect.getdoc(fn) or ''

        is_coroutine = inspect.iscoroutinefunction(fn)
        is_generator = inspect.isgeneratorfunction(fn)
        is_async_generator = inspect.isasyncgenfunction(fn)

        sig = inspect.signature(fn)
        parameters = list(sig.parameters.values())

        if len(parameters) != 1:
            raise ValueError(
                f'Tool \'{name}\' must accept exactly one BaseModel argument.'
            )

        param = parameters[0]
        param_annotation = param.annotation

        if (
            param_annotation is inspect.Parameter.empty
            or not isinstance(param_annotation, type)
            or not issubclass(param_annotation, BaseModel)
        ):
            raise TypeError(
                f'Parameter of tool \'{name}\' must be a Pydantic BaseModel.'
            )

        input_schema = cast(type[P], param_annotation)
        required_fields = [
            fname
            for fname, field in input_schema.model_fields.items()  # type: ignore[attr-defined]
            if field.is_required()
        ]

        return_annotation = sig.return_annotation
        if (
            return_annotation is not inspect.Signature.empty
            and return_annotation is not type(None)  # noqa: E721
            and not is_generator
            and not is_async_generator
        ):
            try:
                from pydantic import create_model

                output_schema = create_model(
                    'ToolOutput',
                    __root__=(return_annotation, ...)
                )
            except Exception:
                output_schema = None
        else:
            output_schema = None

        return cls(
            name=name,
            description=description,
            arg_count=1,
            is_coroutine=is_coroutine,
            is_generator=is_generator,
            is_async_generator=is_async_generator,
            input_schema=input_schema,
            output_schema=output_schema,
            required_fields=required_fields,
        )

    def print_summary(self, stream: TextIO = sys.stdout):

        stream.write('Tool Summary:\n')
        stream.write('-' * 20 + '\n')

        stream.write(f'Name: {self.name}\n')
        stream.write(f'Description: {self.description}\n')
        stream.write(f'Argument Count: {self.arg_count}\n')
        stream.write(f'Is Coroutine: {self.is_coroutine}\n')
        stream.write(f'Is Generator: {self.is_generator}\n')
        stream.write(f'Is Async Generator: {self.is_async_generator}\n')
        stream.write('-' * 20 + '\n')

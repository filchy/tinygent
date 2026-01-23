from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
import inspect
import sys
from typing import Any
from typing import Callable
from typing import Generic
from typing import TextIO
from typing import TypeVar
from typing import cast
from typing import get_type_hints

from pydantic import Field

from tinygent.core.types.base import TinyModel

P = TypeVar('P')
R = TypeVar('R')


@dataclass
class ToolInfo(Generic[R]):
    """Metadata about a tool."""

    name: str
    """The name of the tool."""

    description: str
    """A brief description of the tool's purpose. Extracted from the function's docstring."""

    arg_count: int
    """The number of arguments the tool accepts (should be 1 for TinyModel input)."""

    is_coroutine: bool
    """Indicates if the tool function is a coroutine (async function)."""

    is_generator: bool
    """Indicates if the tool function is a generator (yields values)."""

    is_async_generator: bool
    """Indicates if the tool function is an async generator (yields values asynchronously)."""

    input_schema: type[TinyModel] | None
    """The Pydantic model class representing the input schema for the tool."""

    output_schema: type[TinyModel] | None
    """The Pydantic model class representing the output schema for the tool, if applicable."""

    required_fields: list[str] = field(default_factory=list)
    """List of required fields in the input schema."""

    use_cache: bool = False
    """Indicates if the tool's results should be cached."""

    cache_size: int | None = None
    """The maximum size of the cache, if caching is enabled."""

    uses_auto_schema: bool = False
    """Indicates if the input schema was auto-generated from regular function parameters."""

    @classmethod
    def build_input_model_from_fn(
        cls,
        fn: Callable[..., Any],
    ) -> type[TinyModel] | None:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())

        # zero-arg function results in empty TinyModel
        if not params:
            return type(
                f'{fn.__name__.title()}Input',
                (TinyModel,),
                {},
            )

        if (
            len(params) == 1
            and isinstance(params[0].annotation, type)
            and issubclass(params[0].annotation, TinyModel)
        ):
            return None  # already got TinyModel as single param input so no need to do anything.

        hints = get_type_hints(fn)
        annotations: dict[str, Any] = {}
        namespace: dict[str, Any] = {}

        for p in params:
            if p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise TypeError(f"Tool '{fn.__name__}' cannot use *args or **kwargs")

            annotation = hints.get(p.name, Any)
            default = p.default if p.default is not inspect.Parameter.empty else ...

            annotations[p.name] = annotation
            namespace[p.name] = Field(default)

        namespace['__annotations__'] = annotations

        return type(
            f'{fn.__name__.title()}Input',
            (TinyModel,),
            namespace,
        )

    @property
    def is_cachable(self) -> bool:
        """Indicates if the tool is cachable (not a generator or async generator)."""
        return not (self.is_generator or self.is_async_generator)

    @classmethod
    def from_callable(cls, fn: Callable[..., R], *args, **kwargs) -> ToolInfo[R]:
        """Create a ToolInfo instance from a callable function."""
        name = fn.__name__
        description = inspect.getdoc(fn) or ''

        is_coroutine = inspect.iscoroutinefunction(fn)
        is_generator = inspect.isgeneratorfunction(fn)
        is_async_generator = inspect.isasyncgenfunction(fn)

        sig = inspect.signature(fn)

        input_schema = cls.build_input_model_from_fn(fn)
        uses_auto_schema = input_schema is not None
        if input_schema is None:
            param = next(iter(sig.parameters.values()))
            input_schema = cast(type[TinyModel], param.annotation)

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
                    'ToolOutput', __root__=(return_annotation, ...), __base__=TinyModel
                )
            except Exception:
                output_schema = None
        else:
            output_schema = None

        field_names = {f.name for f in fields(cls)}
        extra_kwargs = {
            key: value for key, value in kwargs.items() if key in field_names
        }

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
            uses_auto_schema=uses_auto_schema,
            **extra_kwargs,
        )

    def print_summary(self, stream: TextIO = sys.stdout):
        """Print a summary of the tool's metadata to the specified stream."""

        stream.write('Tool Summary:\n')
        stream.write('-' * 20 + '\n')

        stream.write(f'Name: {self.name}\n')
        stream.write(f'Description: {self.description}\n')
        stream.write(f'Argument Count: {self.arg_count}\n')
        stream.write(f'Is Coroutine: {self.is_coroutine}\n')
        stream.write(f'Is Generator: {self.is_generator}\n')
        stream.write(f'Is Async Generator: {self.is_async_generator}\n')
        stream.write(f'Input Schema: {self.input_schema}\n')
        stream.write(f'Output Schema: {self.output_schema}\n')
        stream.write(f'Required Fields: {self.required_fields}\n')
        stream.write(f'Use Cache: {self.use_cache}\n')
        if self.use_cache:
            stream.write(f'Cache Size: {self.cache_size}\n')
        stream.write('-' * 20 + '\n')

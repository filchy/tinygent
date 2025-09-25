from dataclasses import replace
from typing import Any
from typing import cast

from pydantic import Field
from pydantic import create_model

from tinygent.datamodels.tool import AbstractTool
from tinygent.types.base import TinyModel


class ToolWithReasoning(AbstractTool):
    def __init__(self, inner_tool: AbstractTool) -> None:
        self._inner = inner_tool
        self._reasoning: str | None = None
        self.__reasoning_field_name = 'reasoning'

        # Dynamically create a new input schema with `reasoning: str`
        original_input = inner_tool.info.input_schema
        if original_input is None:
            raise TypeError('Tool must have an input schema')

        fields = {
            **{k: (v.annotation, v) for k, v in original_input.model_fields.items()},
            self.__reasoning_field_name: (
                str,
                Field(..., description='Why this tool is being called'),
            ),
        }

        self._input_model = create_model(  # type: ignore[call-overload]
            f'{original_input.__name__}WithReasoning',
            __base__=TinyModel,
            **fields,
        )

    @property
    def reasoning(self) -> str:
        if self._reasoning is None:
            raise ValueError('Reasoning has not been set yet.')
        return self._reasoning

    @reasoning.setter
    def reasoning(self, _: str) -> None:
        raise ValueError('Reasoning is read-only and cannot be set directly.')

    @property
    def info(self) -> Any:
        inner_info = self._inner.info
        return replace(inner_info, input_schema=self._input_model)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], dict):
            data = self._input_model(**args[0])
        else:
            data = self._input_model(**kwargs)

        self._reasoning = getattr(cast(Any, data), self.__reasoning_field_name)

        orig_model = self._inner.info.input_schema
        assert orig_model is not None

        input_data = orig_model(
            **{
                k: v
                for k, v in data.model_dump().items()
                if k != self.__reasoning_field_name
            }
        )

        return self._inner(input_data)

    def clear_cache(self) -> None:
        return self._inner.clear_cache()

    def cache_info(self) -> Any:
        return self._inner.cache_info()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

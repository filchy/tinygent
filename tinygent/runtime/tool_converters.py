from typing import Any
from typing import Callable

from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.tool import AbstractTool
from tinygent.llms.openai import OpenAIFunction
from tinygent.llms.openai import OpenAILLM


def tool_converter(
    llm_type: type[AbstractLLM]
) -> Callable[
    [Callable[[AbstractTool], Any]],
    Callable[[AbstractTool], Any]
]:

    from tinygent.runtime.global_registry import GlobalRegistry

    def decorator(
        fn: Callable[[AbstractTool], Any]
    ) -> Callable[[AbstractTool], Any]:

        GlobalRegistry.get_registry().register_tool_convertor(
            llm_type=llm_type,
            fn=fn
        )

        return fn

    return decorator


@tool_converter(llm_type=OpenAILLM)
def openai_tool_convertor(tool: AbstractTool) -> OpenAIFunction:

    info = tool.info
    schema = info.input_schema

    properties: dict[str, OpenAIFunction.FunctionParams.Property] = {}

    if schema:
        for name, field in schema.model_fields.items():
            type_name = (
                field.annotation.__name__
                if isinstance(field.annotation, type)
                else 'string'  # fallback
            )

            properties[name] = OpenAIFunction.FunctionParams.Property(
                type=type_name,
                description=field.description
            )

    return OpenAIFunction(
        type='function',
        name=info.name,
        description=info.description,
        parameters=OpenAIFunction.FunctionParams(
            type='object',
            properties=properties,
            required=info.required_fields
        )
    )

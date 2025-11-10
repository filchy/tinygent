import typing
from typing import cast
from typing import Any
from typing import TypedDict

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import Generation
from google.genai.types import Content
from google.genai.types import FunctionCallingConfigDict
from google.genai.types import FunctionCallingConfigMode
from google.genai.types import ToolConfigDict
from google.genai.types import Tool
from google.genai.chats import ContentOrDict
from google.genai.chats import GenerateContentResponse
from google.genai.types import FunctionCall
from google.genai.types import FunctionResponse
from google.genai.types import GenerateContentConfigDict
from google.genai.types import ModelContent
from google.genai.types import UserContent
from google.genai.types import Part

from tinygent.datamodels.llm_io_result import TinyLLMResult
from tinygent.datamodels.messages import TinyChatMessage
from tinygent.datamodels.messages import TinyToolResult
from tinygent.datamodels.messages import TinyHumanMessage
from tinygent.datamodels.messages import TinyPlanMessage
from tinygent.datamodels.messages import TinyReasoningMessage
from tinygent.datamodels.messages import TinySystemMessage
from tinygent.datamodels.messages import TinyToolCall

if typing.TYPE_CHECKING:
    from tinygent.datamodels.llm_io_input import TinyLLMInput


class GeminiParams(TypedDict):
    message: list[Part]
    history: list[ContentOrDict]


def _gemini_parts_to_text(parts: list[Part] | None) -> str:
    """Convert Gemini Parts to a single text string."""
    texts: list[str] = []
    for part in (parts or []):
        if part.text:
            texts.append(part.text)
    return ''.join(texts)


def _gemini_extract_tool_calls(parts: list[Part] | None) -> list[dict[str, Any]]:
    """Extract tool calls from Gemini Parts."""
    tool_calls: list[dict[str, Any]] = []

    for part in (parts or []):
        if (fc := part.function_call):
            tool_calls.append({
                'id': fc.id,
                'name': fc.name,
                'args': fc.args,
                'type': 'tool_call'
            })
    return tool_calls


def tiny_attributes_to_gemini_config(
    prompt: 'TinyLLMInput',
    temperature: float,
    tools: list[Tool] | None = None,
) -> GenerateContentConfigDict:
    conf_dict: GenerateContentConfigDict = {}
    conf_dict['temperature'] = temperature

    print(f'Tools: {tools}')

    if tools:
        conf_dict['tools'] = tools
        print(f'Setting up Gemini tool config for {len(tools)} tools.')
        conf_dict['tool_config'] = ToolConfigDict(
            function_calling_config=FunctionCallingConfigDict(
                mode=FunctionCallingConfigMode.AUTO
            )
        )

    for msg in prompt.messages:
        if isinstance(msg, TinySystemMessage):
            conf_dict['system_instruction'] = msg.content

    return conf_dict


def tiny_prompt_to_gemini_params(
    prompt: 'TinyLLMInput',
) -> GeminiParams:
    """Convert TinyLLMInput to Gemini GenerateContent parameters."""
    params: list[Content] = []

    for msg in prompt.messages:
        if isinstance(msg, TinyHumanMessage):
            params.append(UserContent(str(msg.content)))

        elif isinstance(msg, TinySystemMessage):
            pass  # INFO: Handled in config_dict

        elif isinstance(msg, TinyChatMessage):
            params.append(ModelContent(str(msg.content)))

        elif isinstance(msg, TinyPlanMessage):
            params.append(
                ModelContent(
                    f'<PLAN>\n{msg.content}\n</PLAN>'
                )
            )

        elif isinstance(msg, TinyReasoningMessage):
            params.append(
                ModelContent(
                    f'<REASONING>\n{msg.content}\n</REASONING>'
                )
            )

        elif isinstance(msg, TinyToolCall):
            params.append(
                ModelContent(
                    parts=[Part(
                        function_call=FunctionCall(
                            id=msg.call_id or 'tool_call_1',
                            name=msg.tool_name,
                            args=msg.arguments
                        )
                    )]
                )
            )

        elif isinstance(msg, TinyToolResult):
            params.append(
                ModelContent(
                    parts=[Part(
                        function_response=FunctionResponse(
                            id=msg.call_id or 'tool_call_1',
                            response={'tool_result': msg.content}
                        )
                    )]
                )
            )

        else:
            raise TypeError(f'Unsupported TinyMessage type: {type(msg)}')

    message = params[-1].parts
    history = params[:-1]

    message = cast(list[Part], message)
    history = cast(list[ContentOrDict], history)

    return GeminiParams(history=history, message=message)


def gemini_response_to_tiny_result(resp: GenerateContentResponse) -> TinyLLMResult:
    """Convert Gemini GenerateContentResponse to TinyLLMResult."""
    generations: list[list[Generation]] = []

    for candidate in (resp.candidates or []):
        if not (content := candidate.content):
            continue

        additional_kwargs = {}

        text = _gemini_parts_to_text(content.parts)
        tool_calls = _gemini_extract_tool_calls(content.parts)

        ai_msg = AIMessage(content=text)

        if tool_calls:
            additional_kwargs['tool_calls'] = tool_calls

            ai_msg.additional_kwargs = additional_kwargs
            ai_msg.tool_calls = cast(list[Any], tool_calls)

        generations.append([ChatGeneration(message=ai_msg, text=text)])

    llm_output = {
        'id': resp.response_id or 'response_id_missing',
        'model': resp.model_version or 'model_missing',
        'created': resp.create_time,
        'usage': resp.usage_metadata.model_dump() if resp.usage_metadata else None,
    }

    return TinyLLMResult(
        generations=generations,
        llm_output=llm_output,
    )

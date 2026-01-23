from __future__ import annotations

import typing
from typing import cast

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import Generation
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionAssistantMessageParam
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionContentPartTextParam
from openai.types.chat import ChatCompletionMessageFunctionToolCall
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import ChatCompletionMessageToolCallUnionParam
from openai.types.chat import ChatCompletionSystemMessageParam
from openai.types.chat import ChatCompletionToolMessageParam
from openai.types.chat import ChatCompletionUserMessageParam

from tinygent.core.datamodels.messages import TinyChatMessage
from tinygent.core.datamodels.messages import TinyChatMessageChunk
from tinygent.core.datamodels.messages import TinyHumanMessage
from tinygent.core.datamodels.messages import TinyPlanMessage
from tinygent.core.datamodels.messages import TinyReasoningMessage
from tinygent.core.datamodels.messages import TinySystemMessage
from tinygent.core.datamodels.messages import TinyToolCall
from tinygent.core.datamodels.messages import TinyToolCallChunk
from tinygent.core.datamodels.messages import TinyToolResult
from tinygent.core.types.io.llm_io_chunks import TinyLLMResultChunk
from tinygent.core.types.io.llm_io_result import TinyLLMResult

if typing.TYPE_CHECKING:
    from tinygent.core.types.io.llm_io_input import TinyLLMInput


def _normalize_content(content: str | list[str | dict]) -> str:
    """Normalize OpenAI ChatCompletion content to a string."""
    if isinstance(content, str):
        return content

    return ''.join(
        part if isinstance(part, str) else f'[{part.get("type", "object")}]'
        for part in content
    )


def _to_text_parts(
    content: object,
) -> str | list[ChatCompletionContentPartTextParam]:
    """Convert content to a string or list of ChatCompletionContentPartTextParam."""
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [{'type': 'text', 'text': str(x)} for x in content]
    return str(content)


def _normalize_tool_calls(
    raw: list[dict] | None,
) -> list[ChatCompletionMessageToolCallUnionParam]:
    """Normalize raw tool calls to ChatCompletionMessageToolCallUnionParam."""
    if not raw:
        return []

    out: list[ChatCompletionMessageToolCallUnionParam] = []
    for tc in raw:
        func = tc.get('function', {})
        out.append(
            cast(
                ChatCompletionMessageToolCallUnionParam,
                {
                    'id': str(tc.get('id', '')),
                    'type': 'function',
                    'function': {
                        'name': str(func.get('name', '')),
                        'arguments': str(func.get('arguments', '')),
                    },
                },
            )
        )
    return out


def tiny_prompt_to_openai_params(
    prompt: 'TinyLLMInput',
) -> list[ChatCompletionMessageParam]:
    """Convert a TinyLLMInput prompt to a list of OpenAI ChatCompletionMessageParam."""
    params: list[ChatCompletionMessageParam] = []

    for msg in prompt.messages:
        if isinstance(msg, TinyHumanMessage):
            params.append(
                ChatCompletionUserMessageParam(role='user', content=msg.content)
            )

        elif isinstance(msg, TinySystemMessage):
            params.append(
                ChatCompletionSystemMessageParam(role='system', content=msg.content)
            )

        elif isinstance(msg, TinyChatMessage):
            params.append(
                ChatCompletionAssistantMessageParam(
                    role='assistant', content=msg.content
                )
            )

        elif isinstance(msg, TinyPlanMessage):
            params.append(
                ChatCompletionAssistantMessageParam(
                    role='assistant', content=f'<PLAN>\n{msg.content}\n</PLAN>'
                )
            )

        elif isinstance(msg, TinyReasoningMessage):
            params.append(
                ChatCompletionAssistantMessageParam(
                    role='assistant',
                    content=f'<REASONING>\n{msg.content}\n</REASONING>',
                )
            )

        elif isinstance(msg, TinyToolCall):
            params.append(
                ChatCompletionAssistantMessageParam(
                    role='assistant',
                    content=None,
                    tool_calls=[
                        {
                            'id': msg.call_id or 'tool_call_1',
                            'type': 'function',
                            'function': {
                                'name': msg.tool_name,
                                'arguments': str(msg.arguments),
                            },
                        }
                    ],
                )
            )

        elif isinstance(msg, TinyToolResult):
            params.append(
                ChatCompletionToolMessageParam(
                    role='tool',
                    content=msg.content,
                    tool_call_id=msg.call_id,
                )
            )

        else:
            raise TypeError(f'Unsupported TinyMessage type: {type(msg)}')

    return params


def openai_result_to_tiny_result(resp: ChatCompletion) -> TinyLLMResult:
    """Convert an OpenAI ChatCompletion response to a TinyLLMResult."""
    generations: list[list[Generation]] = []

    for choice in resp.choices:
        msg = choice.message
        text = msg.content or ''

        additional_kwargs = {}
        if getattr(msg, 'tool_calls', None):
            tool_calls = []
            for tc in msg.tool_calls or []:
                if isinstance(tc, ChatCompletionMessageFunctionToolCall):
                    tool_calls.append(
                        {
                            'id': tc.id,
                            'type': tc.type,
                            'function': {
                                'name': tc.function.name,
                                'arguments': tc.function.arguments,
                            },
                        }
                    )
                else:
                    tool_calls.append({'id': tc.id, 'type': tc.type, 'raw': repr(tc)})
            additional_kwargs['tool_calls'] = tool_calls

        ai_msg = AIMessage(content=text, additional_kwargs=additional_kwargs)
        generations.append([ChatGeneration(message=ai_msg, text=text)])

    llm_output = {
        'id': resp.id,
        'model': resp.model,
        'created': resp.created,
        'usage': resp.usage.model_dump() if resp.usage else None,
        'finish_reasons': [c.finish_reason for c in resp.choices],
    }

    return TinyLLMResult(generations=generations, llm_output=llm_output)


def openai_chunk_to_tiny_chunk(resp_chunk: ChatCompletionChunk) -> TinyLLMResultChunk:
    """Convert an OpenAI ChatCompletionChunk to a TinyLLMResultChunk."""
    choice = resp_chunk.choices[0]
    delta = choice.delta

    # tool call chunk
    if delta.tool_calls:
        for tc in delta.tool_calls:
            if tc.function:  # no need for isinstance check
                return TinyLLMResultChunk(
                    type='tool_call',
                    tool_call=TinyToolCallChunk(
                        tool_name=tc.function.name or '',
                        arguments=tc.function.arguments or '',
                        call_id=tc.id or None,
                        index=tc.index,
                        metadata={'raw': tc.model_dump()},
                    ),
                    metadata={'finish_reason': choice.finish_reason},
                )

    # text chunk
    if delta.content:
        return TinyLLMResultChunk(
            type='message',
            message=TinyChatMessageChunk(
                content=_normalize_content(delta.content),
                metadata={'raw': delta.model_dump()},
            ),
            metadata={'finish_reason': choice.finish_reason},
        )

    # end of stream
    if choice.finish_reason is not None:
        return TinyLLMResultChunk(
            type='end',
            metadata={'finish_reason': choice.finish_reason},
        )

    return TinyLLMResultChunk(type='none', metadata={'raw': delta.model_dump()})

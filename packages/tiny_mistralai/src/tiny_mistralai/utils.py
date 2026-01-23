import typing
from typing import cast

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import Generation
from mistralai import UNSET
from mistralai import AssistantMessage
from mistralai import ChatCompletionResponse
from mistralai import CompletionChunk
from mistralai import Content
from mistralai import FunctionCall
from mistralai import SystemMessage
from mistralai import TextChunk
from mistralai import ToolCall
from mistralai import ToolMessage
from mistralai import UserMessage

from tiny_mistralai.types import ChatCompletionMessageParams
from tinygent.core.datamodels.messages import TinyChatMessage
from tinygent.core.datamodels.messages import TinyChatMessageChunk
from tinygent.core.datamodels.messages import TinyHumanMessage
from tinygent.core.datamodels.messages import TinyPlanMessage
from tinygent.core.datamodels.messages import TinyReasoningMessage
from tinygent.core.datamodels.messages import TinySystemMessage
from tinygent.core.datamodels.messages import TinyToolCall
from tinygent.core.datamodels.messages import TinyToolResult
from tinygent.core.types.io.llm_io_chunks import TinyLLMResultChunk
from tinygent.core.types.io.llm_io_chunks import TinyToolCallChunk
from tinygent.core.types.io.llm_io_result import TinyLLMResult

if typing.TYPE_CHECKING:
    from tinygent.core.types.io.llm_io_input import TinyLLMInput


def _normalize_content(content: Content) -> str:
    """Normalize Mistral AI Content to a string."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return ''.join(chunk.text for chunk in content if isinstance(chunk, TextChunk))
    else:
        return ''


def tiny_prompt_to_mistralai_params(
    prompt: 'TinyLLMInput',
) -> list[ChatCompletionMessageParams]:
    params: list[ChatCompletionMessageParams] = []

    for msg in prompt.messages:
        if isinstance(msg, TinyHumanMessage):
            params.append(UserMessage(role='user', content=str(msg.content)))

        elif isinstance(msg, TinySystemMessage):
            params.append(SystemMessage(role='system', content=str(msg.content)))

        elif isinstance(msg, TinyChatMessage):
            params.append(AssistantMessage(role='assistant', content=msg.content))

        elif isinstance(msg, TinyPlanMessage):
            params.append(
                AssistantMessage(
                    role='assistant', content=f'<PLAN>\n{msg.content}\n</PLAN>'
                )
            )

        elif isinstance(msg, TinyReasoningMessage):
            params.append(
                AssistantMessage(
                    role='assistant',
                    content=f'<REASONING>\n{msg.content}\n</REASONING>',
                )
            )

        elif isinstance(msg, TinyToolCall):
            params.append(
                AssistantMessage(
                    role='assistant',
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=msg.call_id or 'tool_call_1',
                            type='function',
                            function=FunctionCall(
                                name=msg.tool_name, arguments=str(msg.arguments)
                            ),
                        )
                    ],
                )
            )

        elif isinstance(msg, TinyToolResult):
            params.append(
                ToolMessage(
                    role='tool',
                    content=msg.content,
                    tool_call_id=msg.call_id,
                )
            )

        else:
            raise TypeError(f'Unsupported TinyMessage type: {type(msg)}')

    return params


def mistralai_result_to_tiny_result(resp: ChatCompletionResponse) -> TinyLLMResult:
    """Convert a Mistral AI ChatCompletionResponse to a TinyLLMResult."""
    generations: list[list[Generation]] = []

    for choice in resp.choices:
        msg = choice.message
        text = _normalize_content(msg.content) if msg.content else ''

        additional_kwargs = {}
        if getattr(msg, 'tool_calls', None):
            tool_calls = []
            for tc in msg.tool_calls or []:
                tool_calls.append(
                    {
                        'id': tc.id,
                        'type': 'function',
                        'function': {
                            'name': tc.function.name,
                            'arguments': tc.function.arguments,
                        },
                    }
                )
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


def mistralai_chunk_to_tiny_chunk(chunk: CompletionChunk) -> TinyLLMResultChunk:
    """Convert a Mistral AI CompletionChunk to a TinyLLMResultChunk."""
    choice = chunk.choices[0]
    delta = choice.delta

    # tool call chunk
    if (tcs := delta.tool_calls) is not UNSET and tcs is not None:
        for tc in cast(list[ToolCall], tcs):
            if tc.function:
                return TinyLLMResultChunk(
                    type='tool_call',
                    tool_call=TinyToolCallChunk(
                        tool_name=tc.function.name or '',
                        arguments=str(tc.function.arguments) or '',
                        call_id=tc.id,
                        index=tc.index or 0,
                        metadata={'raw': tc.model_dump()},
                    ),
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


def mistralai_chunk_to_tiny_chunks(chunk: CompletionChunk) -> list[TinyLLMResultChunk]:
    """Convert a Mistral AI CompletionChunk to a list of TinyLLMResultChunks."""
    choice = chunk.choices[0]
    delta = choice.delta

    results: list[TinyLLMResultChunk] = []

    if (tcs := delta.tool_calls) is not UNSET and tcs is not None:
        for tc in cast(list[ToolCall], tcs):
            if tc.function:
                results.append(
                    TinyLLMResultChunk(
                        type='tool_call',
                        tool_call=TinyToolCallChunk(
                            tool_name=tc.function.name or '',
                            arguments=str(tc.function.arguments) or '',
                            call_id=tc.id,
                            index=tc.index or 0,
                            metadata={'raw': tc.model_dump()},
                        ),
                    )
                )

    if delta.content:
        results.append(
            TinyLLMResultChunk(
                type='message',
                message=TinyChatMessageChunk(
                    content=_normalize_content(delta.content),
                    metadata={'raw': delta.model_dump()},
                ),
                metadata={'finish_reason': choice.finish_reason},
            )
        )

    if choice.finish_reason is not None:
        results.append(
            TinyLLMResultChunk(
                type='end',
                metadata={'finish_reason': choice.finish_reason},
            )
        )

    if not results:
        results.append(
            TinyLLMResultChunk(type='none', metadata={'raw': delta.model_dump()})
        )

    return results


def mistralai_family_to_tokenizer(model: str) -> str:
    m = model.lower()

    if m.startswith('codestral'):
        return 'mistralai/Codestral-22B-v0.1'

    if m.startswith('ministral'):
        return 'mistralai/Ministral-8B-Instruct-2410'

    if m.startswith('pixtral'):
        return 'mistralai/Pixtral-12B-2409'

    if m.startswith('devstral'):
        return 'mistralai/Mistral-7B-Instruct-v0.2'

    if m.startswith('magistral'):
        if 'small' in m:
            return 'mistralai/Mistral-7B-Instruct-v0.2'
        return 'mistralai/Mixtral-8x7B-Instruct-v0.1'

    if m.startswith('mistral'):
        if 'large' in m:
            return 'mistralai/Mixtral-8x22B-Instruct-v0.1'
        if 'medium' in m:
            return 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        return 'mistralai/Mistral-7B-Instruct-v0.2'

    return 'mistralai/Mistral-7B-Instruct-v0.2'

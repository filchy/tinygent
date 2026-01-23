import typing

from anthropic.types import Message
from anthropic.types import MessageParam
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import Generation

from tinygent.core.datamodels.messages import TinyChatMessage
from tinygent.core.datamodels.messages import TinyHumanMessage
from tinygent.core.datamodels.messages import TinyPlanMessage
from tinygent.core.datamodels.messages import TinyReasoningMessage
from tinygent.core.datamodels.messages import TinySystemMessage
from tinygent.core.datamodels.messages import TinyToolCall
from tinygent.core.datamodels.messages import TinyToolResult
from tinygent.core.types.io.llm_io_chunks import TinyChatMessageChunk
from tinygent.core.types.io.llm_io_chunks import TinyLLMResultChunk
from tinygent.core.types.io.llm_io_result import TinyLLMResult

if typing.TYPE_CHECKING:
    from tinygent.core.types.io.llm_io_input import TinyLLMInput


def tiny_prompt_to_anthropic_params(
    prompt: 'TinyLLMInput',
) -> tuple[list[MessageParam], str | None]:
    """Convert a TinyLLMInput prompt to a list of Anthropic MessageParam."""
    params: list[MessageParam] = []
    system_message: str | None = None

    for msg in prompt.messages:
        if isinstance(msg, TinyHumanMessage):
            params.append(MessageParam(role='user', content=msg.content))

        elif isinstance(msg, TinySystemMessage):
            # system message is separate param
            system_message = msg.content

        elif isinstance(msg, TinyChatMessage):
            params.append(MessageParam(role='assistant', content=msg.content))

        elif isinstance(msg, TinyPlanMessage):
            params.append(
                MessageParam(
                    role='assistant',
                    content=[
                        {
                            'type': 'redacted_thinking',
                            'data': f'<PLAN>\n{msg.content}\n</PLAN>',
                        }
                    ],
                )
            )

        elif isinstance(msg, TinyReasoningMessage):
            params.append(
                MessageParam(
                    role='assistant',
                    content=[
                        {
                            'type': 'redacted_thinking',
                            'data': f'<REASONING>\n{msg.content}\n</REASONING>',
                        }
                    ],
                )
            )

        elif isinstance(msg, TinyToolCall):
            params.append(
                {
                    'role': 'assistant',
                    'content': [
                        {
                            'type': 'tool_use',
                            'id': msg.call_id or 'tool_call_1',
                            'name': msg.tool_name,
                            'input': msg.arguments,
                        }
                    ],
                }
            )

        elif isinstance(msg, TinyToolResult):
            params.append(
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'tool_result',
                            'tool_use_id': msg.call_id,
                            'content': msg.content,
                        }
                    ],
                }
            )

        else:
            raise TypeError(f'Unsupported TinyMessage type: {type(msg)}')

    return params, system_message


def anthropic_result_to_tiny_result(resp: Message) -> TinyLLMResult:
    """Convert an Anthropic Message response to a TinyLLMResult."""
    generations: list[list[Generation]] = []

    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in resp.content:
        if block.type == 'text':
            text_parts.append(block.text)

        elif block.type == 'tool_use':
            tool_calls.append(
                {
                    'id': block.id,
                    'name': block.name,
                    'args': block.input,
                }
            )

    text = ''.join(text_parts)

    additional_kwargs = {}
    if tool_calls:
        additional_kwargs['tool_calls'] = tool_calls

    ai_msg = AIMessage(
        content=text, additional_kwargs=additional_kwargs, **additional_kwargs
    )
    generations.append([ChatGeneration(message=ai_msg, text=text)])

    llm_output = {
        'id': resp.id,
        'model': resp.model,
        'stop_reason': resp.stop_reason,
        'usage': resp.usage.model_dump() if resp.usage else None,
    }

    return TinyLLMResult(generations=generations, llm_output=llm_output)


def anthropic_chunk_to_tiny_chunk(chunk: str) -> TinyLLMResultChunk:
    """Convert an Anthropic ChatCompletionChunk to a TinyLLMResultChunk."""
    return TinyLLMResultChunk(
        type='message',
        message=TinyChatMessageChunk(
            content=chunk,
        ),
    )

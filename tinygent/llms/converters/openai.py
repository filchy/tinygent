from typing import cast
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import LLMResult
from langchain_core.outputs import Generation
from langchain_core.prompt_values import PromptValue
from langchain_core.messages.utils import convert_to_openai_messages
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionMessageFunctionToolCall
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import ChatCompletionSystemMessageParam
from openai.types.chat import ChatCompletionUserMessageParam
from openai.types.chat import ChatCompletionAssistantMessageParam
from openai.types.chat import ChatCompletionToolMessageParam
from openai.types.chat import ChatCompletionContentPartTextParam
from openai.types.chat import ChatCompletionMessageToolCallUnionParam


def _to_text_parts(
    content: object,
) -> str | list[ChatCompletionContentPartTextParam]:

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


def lc_prompt_to_openai_params(
    prompt: PromptValue
) -> list[ChatCompletionMessageParam]:

    raw = convert_to_openai_messages(prompt.to_messages())
    params: list[ChatCompletionMessageParam] = []

    for m in raw:
        role = m['role']
        content = _to_text_parts(m.get('content'))

        if role == 'system':
            params.append(
                ChatCompletionSystemMessageParam(
                    role='system',
                    content=content
                )
            )

        elif role == 'user':
            params.append(
                ChatCompletionUserMessageParam(
                    role='user',
                    content=content
                )
            )

        elif role == 'assistant':
            tool_calls = _normalize_tool_calls(m.get('tool_calls'))
            params.append(
                ChatCompletionAssistantMessageParam(
                    role='assistant',
                    content=content,
                    tool_calls=tool_calls,
                )
            )

        elif role == 'tool':
            params.append(
                ChatCompletionToolMessageParam(
                    role='tool',
                    content=content,
                    tool_call_id=m.get('tool_call_id', ''),
                )
            )

        else:
            raise TypeError(f'Unsupported role from LC: {role!r}')

    return params


def openai_result_to_lc_result(resp: ChatCompletion) -> LLMResult:
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
                    tool_calls.append(
                        {'id': tc.id, 'type': tc.type, 'raw': tc}
                    )
            additional_kwargs['tool_calls'] = tool_calls

        ai_msg = AIMessage(content=text, additional_kwargs=additional_kwargs)
        generations.append([ChatGeneration(message=ai_msg, text=text)])

    llm_output = {
        'id': resp.id,
        'model': resp.model,
        'created': resp.created,
        'usage': resp.usage.dict() if resp.usage else None,
        'finish_reasons': [c.finish_reason for c in resp.choices],
    }

    return LLMResult(generations=generations, llm_output=llm_output)


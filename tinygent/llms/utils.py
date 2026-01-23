from __future__ import annotations

from collections import defaultdict
from io import StringIO
import json
from typing import Any
from typing import AsyncIterator

from tinygent.core.datamodels.messages import TinyToolCall
from tinygent.core.types.io.llm_io_chunks import TinyLLMResultChunk


def group_chunks_for_telemetry(chunks: list[TinyLLMResultChunk]) -> list[str]:
    """Group chunks into meaningful telemetry entries.

    Groups consecutive text chunks into single entries and
    groups tool call chunks by their complete tool calls.

    Args:
        chunks: List of TinyLLMResultChunk objects from streaming.

    Returns:
        List of grouped string representations for telemetry.
    """
    if not chunks:
        return []

    grouped: list[str] = []
    text_buffer = StringIO()
    current_tool_calls: dict[int, dict[str, str]] = {}

    for chunk in chunks:
        if chunk.type == 'message' and chunk.message:
            # Accumulate text content
            if chunk.message.content:
                text_buffer.write(chunk.message.content)

        elif chunk.type == 'tool_call' and chunk.tool_call:
            # Flush text buffer before processing tool calls
            if text_content := text_buffer.getvalue():
                grouped.append(f'text: {text_content}')
                text_buffer = StringIO()

            # Accumulate tool call parts by index
            tc = chunk.tool_call
            if tc.index not in current_tool_calls:
                current_tool_calls[tc.index] = {'name': '', 'arguments': ''}
            if tc.tool_name:
                current_tool_calls[tc.index]['name'] += tc.tool_name
            if tc.arguments:
                current_tool_calls[tc.index]['arguments'] += tc.arguments

        elif chunk.type == 'tool_call' and chunk.full_tool_call:
            # Full tool call is already complete
            ftc = chunk.full_tool_call
            grouped.append(f'tool_call: {ftc.tool_name}({ftc.arguments})')

    # Flush remaining text
    if text_content := text_buffer.getvalue():
        grouped.append(f'text: {text_content}')

    # Flush accumulated tool calls
    for _idx, tc_data in sorted(current_tool_calls.items()):
        if tc_data['name']:
            grouped.append(f'tool_call: {tc_data["name"]}({tc_data["arguments"]})')

    return grouped if grouped else ['empty_response']


def accumulate_llm_chunks(
    tiny_chunks: AsyncIterator[TinyLLMResultChunk],
) -> AsyncIterator[TinyLLMResultChunk]:
    """Generic accumulator that merges partial tool call chunks into complete TinyToolCall objects."""
    pending_tool_calls: dict[int, dict[str, Any]] = defaultdict(
        lambda: {'id': '', 'name': '', 'args': []}
    )

    async def _gen():
        async for tiny_chunk in tiny_chunks:
            if tiny_chunk.is_message and tiny_chunk.message:
                yield tiny_chunk

            elif tiny_chunk.is_tool_call and tiny_chunk.tool_call:
                tc = tiny_chunk.tool_call
                state = pending_tool_calls[tc.index]

                if tc.call_id and not state['id']:
                    state['id'] = tc.call_id
                if tc.tool_name and not state['name']:
                    state['name'] = tc.tool_name

                if tc.arguments:
                    state['args'].append(tc.arguments)

                    try:
                        args_str = ''.join(state['args'])
                        tool_args = json.loads(args_str)
                    except json.JSONDecodeError:
                        continue  # wait for more chunks

                    yield TinyLLMResultChunk(
                        type='tool_call',
                        full_tool_call=TinyToolCall(
                            tool_name=state['name'],
                            arguments=tool_args,
                            call_id=state['id'] or None,
                            metadata={'raw': state},
                        ),
                        metadata=tiny_chunk.metadata,
                    )

            elif tiny_chunk.is_end:
                yield tiny_chunk

    return _gen()

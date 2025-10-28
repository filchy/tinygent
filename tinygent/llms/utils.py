from collections import defaultdict
import json
from typing import Any
from typing import AsyncIterator

from tinygent.datamodels.llm_io_chunks import TinyLLMResultChunk
from tinygent.datamodels.messages import TinyToolCall


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

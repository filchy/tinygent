from itertools import chain
from typing import Iterator
from typing import Literal
from typing import cast
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import LLMResult
from pydantic import BaseModel

from tinygent.llms.utils import normalize_content


class TinyChatMessage(BaseModel):

    type: Literal['chat'] = 'chat'

    content: str

    metadata: dict = {}


class TinyToolCall(BaseModel):

    type: Literal['tool'] = 'tool'

    tool_name: str

    arguments: dict

    call_id: str | None = None

    metadata: dict = {}


MessageType = TinyChatMessage | TinyToolCall


class TinyLLMResult(LLMResult):

    def tiny_iter(self) -> Iterator[MessageType]:

        for generation in chain.from_iterable(self.generations):
            chat_gen = cast(ChatGeneration, generation)
            message = chat_gen.message

            if not isinstance(message, AIMessage):
                raise ValueError('Unsupported message type %s' % type(message))

            if (tool_calls := message.tool_calls):
                for tool_call in tool_calls:
                    yield TinyToolCall(
                        tool_name=tool_call['name'],
                        arguments=tool_call['args'],
                        call_id=tool_call['id'] or None,
                        metadata={'raw': tool_call}
                    )
            elif (content := message.content):
                yield TinyChatMessage(
                    content=normalize_content(content),
                    metadata={'raw': message}
                )

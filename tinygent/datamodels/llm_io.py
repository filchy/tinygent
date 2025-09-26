from itertools import chain
from typing import Iterator
from typing import cast

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import LLMResult
from openai import BaseModel

from tinygent.datamodels.messages import AllTinyMessages
from tinygent.datamodels.messages import TinyAIMessage
from tinygent.datamodels.messages import TinyChatMessage
from tinygent.datamodels.messages import TinyToolCall
from tinygent.llms.utils import normalize_content


class TinyLLMInput(BaseModel):
    """Input to an LLM, consisting of a list of messages."""

    messages: list[AllTinyMessages]


class TinyLLMResult(LLMResult):
    """Result from an LLM, consisting of generations and optional metadata."""

    def tiny_iter(self) -> Iterator[TinyAIMessage]:
        """Iterate over the messages and tool calls in the LLM result."""
        for generation in chain.from_iterable(self.generations):
            chat_gen = cast(ChatGeneration, generation)
            message = chat_gen.message

            if not isinstance(message, AIMessage):
                raise ValueError('Unsupported message type %s' % type(message))

            if tool_calls := message.tool_calls:
                for tool_call in tool_calls:
                    yield TinyToolCall(
                        tool_name=tool_call['name'],
                        arguments=tool_call['args'],
                        call_id=tool_call['id'] or None,
                        metadata={'raw': tool_call},
                    )
            elif content := message.content:
                yield TinyChatMessage(
                    content=normalize_content(content), metadata={'raw': message}
                )

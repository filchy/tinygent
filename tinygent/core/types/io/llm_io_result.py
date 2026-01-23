from itertools import chain
from typing import Iterator
from typing import cast

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import LLMResult

from tinygent.core.datamodels.messages import TinyAIMessage
from tinygent.core.datamodels.messages import TinyChatMessage
from tinygent.core.datamodels.messages import TinyToolCall


class TinyLLMResult(LLMResult):
    """Result from an LLM, consisting of generations and optional metadata."""

    @staticmethod
    def normalize_content(content: str | list[str | dict]) -> str:
        """Normalize content to a string."""
        if isinstance(content, str):
            return content

        return ''.join(
            part if isinstance(part, str) else f'[{part.get("type", "object")}]'
            for part in content
        )

    def to_string(self) -> str:
        return '\n'.join(
            msg.content for msg in self.tiny_iter() if isinstance(msg, TinyChatMessage)
        )

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

            if content := message.content:
                yield TinyChatMessage(
                    content=self.normalize_content(content), metadata={'raw': message}
                )

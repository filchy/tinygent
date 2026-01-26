from io import StringIO
from typing import Literal

from pydantic import Field

from tinygent.core.datamodels.llm import AbstractLLM
from tinygent.core.datamodels.llm import AbstractLLMConfig
from tinygent.core.datamodels.memory import AbstractMemoryConfig
from tinygent.core.datamodels.messages import AllTinyMessages
from tinygent.core.datamodels.messages import TinyHumanMessage
from tinygent.core.datamodels.messages import TinySummaryMessage
from tinygent.core.datamodels.messages import TinySystemMessage
from tinygent.core.factory.llm import build_llm
from tinygent.core.prompts.memory.factory.buffer_summary_chat_memory import (
    get_prompt_template,
)
from tinygent.core.prompts.memory.template.buffer_summary_chat_memory import (
    SummaryUpdatePromptTemplate,
)
from tinygent.core.types.io.llm_io_input import TinyLLMInput
from tinygent.memory.base_chat_memory import BaseChatMemory
from tinygent.utils.jinja_utils import render_template

_DEFAULT_PROMPT = get_prompt_template()


class BufferSummaryChatMemoryConfig(AbstractMemoryConfig['BufferSummaryChatMemory']):
    type: Literal['buffer_summary'] = Field(default='buffer_summary', frozen=True)

    max_token_limit: int = Field(default=2000)

    return_messages: bool = Field(default=False)

    llm: AbstractLLM | AbstractLLMConfig = Field(...)

    prompt: SummaryUpdatePromptTemplate = Field(default=_DEFAULT_PROMPT)

    def build(self) -> 'BufferSummaryChatMemory':
        return BufferSummaryChatMemory(
            llm=self.llm if isinstance(self.llm, AbstractLLM) else build_llm(self.llm),
            max_token_limit=self.max_token_limit,
            return_messages=self.return_messages,
            prompt=self.prompt,
        )


class BufferSummaryChatMemory(BaseChatMemory):
    """Intelligent memory that summarizes old messages to save tokens.

    Maintains recent messages in full while summarizing older messages that
    would exceed the token limit. Uses an LLM to create progressive summaries
    that capture key information from pruned messages.

    The memory automatically prunes messages when the buffer exceeds the token
    limit, creating or updating a rolling summary. Recent messages stay intact
    for immediate context while older content is compressed into summaries.

    Suitable for:
    - Long conversations requiring full context awareness
    - Token-limited environments needing history retention
    - Scenarios where both recent details and historical context matter

    Args:
        llm: Language model for generating summaries
        max_token_limit: Maximum tokens before triggering summarization (default: 2000)
        return_messages: Return messages as objects vs strings (default: False)
        prompt: Template for summary generation (default provided)
    """

    def __init__(
        self,
        llm: AbstractLLM,
        max_token_limit: int = 2000,
        return_messages: bool = False,
        prompt: SummaryUpdatePromptTemplate = _DEFAULT_PROMPT,
    ) -> None:
        super().__init__()

        self.llm = llm
        self.max_token_limit = max_token_limit
        self.return_messages = return_messages
        self.prompt = prompt

        self._memory_key: str = 'summarized_chat'

        self._summary_message: TinySummaryMessage | None = None

    @property
    def memory_keys(self) -> list[str]:
        return [self._memory_key]

    def load_variables(self) -> dict[str, str | list[AllTinyMessages]]:
        final_buffer = self._chat_history.messages
        if self._summary_message:
            final_buffer.insert(0, self._summary_message)

        final: str | list[AllTinyMessages]
        if self.return_messages:
            final = final_buffer
        else:
            final = '\n'.join([m.tiny_str for m in final_buffer])

        return {
            self._memory_key: final,
        }

    def save_context(self, message: AllTinyMessages) -> None:
        super().save_context(message)
        self.prune()

    def prune(self) -> None:
        curr_buffer_memory = self._chat_history.messages
        curr_buffer_length = self.llm.count_tokens_in_messages(curr_buffer_memory)

        if curr_buffer_length > self.max_token_limit:
            pruned_buffer_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_buffer_memory.append(curr_buffer_memory.pop(0))
                curr_buffer_length = self.llm.count_tokens_in_messages(
                    curr_buffer_memory
                )

            summary_text = self.llm.generate_text(
                llm_input=TinyLLMInput(
                    messages=[
                        TinySystemMessage(content=self.prompt.system),
                        TinyHumanMessage(
                            content=render_template(
                                self.prompt.user,
                                {
                                    'summary': self._summary_message.content
                                    if self._summary_message
                                    else '',
                                    'new_lines': '\n'.join(
                                        [m.tiny_str for m in pruned_buffer_memory]
                                    ),
                                },
                            )
                        ),
                    ]
                )
            )
            self._summary_message = TinySummaryMessage(content=summary_text.to_string())

    def __str__(self) -> str:
        base = super().__str__()

        buff = StringIO()

        buff.write(base)
        buff.write('\ttype: Buffer Summary Memory\n')
        buff.write(f'\tMax token limit: {self.max_token_limit}')

        return buff.getvalue()

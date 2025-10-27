from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import logging
import typing
from typing import Literal

from tinygent.agents.base_agent import TinyBaseAgent
from tinygent.agents.base_agent import TinyBaseAgentConfig
from tinygent.cli.builder import build_llm
from tinygent.cli.builder import build_tool
from tinygent.datamodels.llm_io_chunks import TinyLLMResultChunk
from tinygent.datamodels.llm_io_input import TinyLLMInput
from tinygent.datamodels.memory import AbstractMemory
from tinygent.datamodels.messages import TinyChatMessage
from tinygent.datamodels.messages import TinyChatMessageChunk
from tinygent.datamodels.messages import TinyHumanMessage
from tinygent.datamodels.messages import TinyReasoningMessage
from tinygent.datamodels.messages import TinyToolCall
from tinygent.memory.buffer_chat_memory import BufferChatMemory
from tinygent.tools.reasoning_tool import ReasoningTool
from tinygent.types.base import TinyModel
from tinygent.utils import render_template

if typing.TYPE_CHECKING:
    from tinygent.datamodels.llm import AbstractLLM
    from tinygent.datamodels.tool import AbstractTool

logger = logging.getLogger(__name__)


class ReasonPromptTemplate(TinyModel):
    """Used to define the reasoning step."""

    init: str
    update: str

    _template_fields = {'init': {'task'}, 'update': {'task', 'overview'}}


class ActionPromptTemplate(TinyModel):
    """Used to define the final answer or action."""

    action: str

    _template_fields = {'action': {'reasoning', 'tools'}}


class FallbackPromptTemplate(TinyModel):
    """Used to define the fallback if agent don't answer in time."""

    fallback_answer: str

    _template_fields = {'fallback_answer': {'task', 'overview'}}


class ReActPromptTemplate(TinyModel):
    """Prompt template for ReAct Agent."""

    reason: ReasonPromptTemplate
    action: ActionPromptTemplate
    fallback: FallbackPromptTemplate


class TinyReActAgentConfig(TinyBaseAgentConfig['TinyReActAgent']):
    """Configuration for ReAct Agent."""

    type: Literal['react'] = 'react'

    prompt_template: ReActPromptTemplate
    max_iterations: int = 10

    def build(self) -> TinyReActAgent:
        return TinyReActAgent(
            llm=build_llm(self.llm),
            prompt_template=self.prompt_template,
            tools=[build_tool(tool) for tool in self.tools],
            max_iterations=self.max_iterations,
        )


class TinyReActAgent(TinyBaseAgent):
    """ReAct Agent implementation."""

    def __init__(
        self,
        llm: AbstractLLM,
        prompt_template: ReActPromptTemplate,
        tools: list[AbstractTool] = [],
        memory: AbstractMemory = BufferChatMemory(),
        max_iterations: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(llm=llm, tools=tools, memory=memory, **kwargs)

        class TinyReactIteration(TinyModel):
            iteration_number: int
            tool_calls: list[TinyToolCall]
            reasoning: str

            @property
            def summary(self) -> str:
                return (
                    f'Iteration {self.iteration_number}:\n'
                    f'Reasoning: {self.reasoning}\n'
                    f'Tool Calls: {", ".join(call.tool_name for call in self.tool_calls)}\n'
                )

        self.TinyReactIteration = TinyReactIteration

        self._iteration_number: int = 1
        self._react_iterations: list[TinyReactIteration] = []

        self.prompt_template = prompt_template
        self.max_iterations = max_iterations

        self.memory = memory

    def _stream_reasoning(self, task: str) -> TinyChatMessage | TinyReasoningMessage:
        class TinyReasoningOutcome(TinyModel):
            type: Literal['reasoning', 'final_answer']
            content: str

        if self._iteration_number == 1:
            template = self.prompt_template.reason.init
            variables = {'task': task}
        else:
            template = self.prompt_template.reason.update
            variables = {
                'task': task,
                'overview': '\n'.join(
                    iteration.summary for iteration in self._react_iterations
                ),
            }

        messages = TinyLLMInput(
            messages=[
                *self.memory.chat_messages,
                TinyHumanMessage(content=render_template(template, variables)),
            ]
        )

        result = self.run_llm(
            fn=self.llm.generate_structured,
            llm_input=messages,
            output_schema=TinyReasoningOutcome,
        )

        if result.type == 'final_answer':
            return TinyChatMessage(content=result.content)
        return TinyReasoningMessage(content=result.content)

    async def _stream_action(self, reasoning: str) -> AsyncGenerator[TinyToolCall]:
        messages = TinyLLMInput(
            messages=[
                *self.memory.chat_messages,
                TinyHumanMessage(
                    content=render_template(
                        self.prompt_template.action.action,
                        {'reasoning': reasoning, 'tools': self._tools},
                    )
                ),
            ]
        )

        async for chunk in self.run_llm_stream(
            fn=self.llm.stream_with_tools,
            llm_input=messages,
            tools=self._tools,
        ):
            if chunk.is_tool_call and chunk.full_tool_call:
                yield chunk.full_tool_call

    async def _stream_fallback(self, task: str) -> AsyncGenerator[str]:
        messages = TinyLLMInput(
            messages=[
                *self.memory.chat_messages,
                TinyHumanMessage(
                    content=render_template(
                        self.prompt_template.fallback.fallback_answer,
                        {
                            'task': task,
                            'overview': '\n'.join(
                                iteration.summary for iteration in self._react_iterations
                            ),
                        },
                    )
                ),
            ]
        )

        async for chunk in self.run_llm_stream(
            fn=self.llm.stream_text,
            llm_input=messages,
        ):
            if isinstance(chunk, TinyLLMResultChunk) and chunk.is_message:
                assert isinstance(chunk.message, TinyChatMessageChunk)
                yield chunk.message.content

    async def _run_agent(self, input_text: str) -> AsyncGenerator[str]:
        self._iteration_number = 1
        returned_final_answer: bool = False

        self.memory.save_context(TinyHumanMessage(content=input_text))

        while not returned_final_answer and (
            self._iteration_number <= self.max_iterations
        ):
            logger.debug('--- ITERATION %d ---', self._iteration_number)

            try:
                reasoning_result = self._stream_reasoning(task=input_text)
                logger.debug(
                    '[%d. ITERATION - Reasoning Result]: %s',
                    self._iteration_number,
                    reasoning_result.content,
                )

                if isinstance(reasoning_result, TinyChatMessage):
                    logger.debug(
                        '[%d. ITERATION - Reasoning Final Answer]: %s',
                        self._iteration_number,
                        reasoning_result.content,
                    )
                    returned_final_answer = True

                    self.memory.save_context(reasoning_result)
                    self.on_answer(reasoning_result.content)

                    yield reasoning_result.content

                else:
                    logger.debug(
                        '[%d. ITERATION - Streaming Action]', self._iteration_number
                    )

                    tool_calls: list[TinyToolCall] = []
                    async for msg in self._stream_action(
                        reasoning=reasoning_result.content
                    ):
                        called_tool = self.get_tool(msg.tool_name)
                        if called_tool:
                            tool_result = self.run_tool(called_tool, msg)

                            self.memory.save_context(msg)
                            self.memory.save_context(tool_result)

                            tool_calls.append(msg)

                            if isinstance(called_tool, ReasoningTool):
                                reasoning = msg.arguments.get('reasoning', '')
                                logger.debug(
                                    '[%d. ITERATION - Tool Reasoning]: %s',
                                    self._iteration_number,
                                    reasoning,
                                )
                                self.on_tool_reasoning(reasoning)
                        else:
                            logger.error(
                                'Tool %s not found. Skipping tool call.', msg.tool_name
                            )

                        logger.debug(
                            '[%s. ITERATION - Tool Call]: %s(%s) = %s',
                            self._iteration_number,
                            msg.tool_name,
                            msg.arguments,
                            msg.result,
                        )

                    self._react_iterations.append(
                        self.TinyReactIteration(
                            iteration_number=self._iteration_number,
                            tool_calls=tool_calls,
                            reasoning=reasoning_result.content,
                        )
                    )
            except Exception as e:
                self.on_error(e)
                raise e
            finally:
                self._iteration_number += 1

        if not returned_final_answer:
            logger.warning(
                'Max iterations reached without final answer. Using fallback.'
                'Returning fallback answer.'
            )

            yielded_final_answer = False
            final_yielded_answer = ''

            async for fallback_chunk in self._stream_fallback(task=input_text):
                yielded_final_answer = True
                final_yielded_answer += fallback_chunk

                yield fallback_chunk

            if not yielded_final_answer:
                final_yielded_answer = 'I have completed my reasoning and tool usage but did not arrive at a final answer.'
                yield final_yielded_answer

            self.memory.save_context(TinyChatMessage(content=final_yielded_answer))
            self.on_answer(final_yielded_answer)

    def run(
        self,
        input_text: str,
        reset: bool = True,
    ) -> str:
        logger.debug('[USER INPUT] %s', input_text)

        if reset:
            logger.debug('[AGENT RESET]')

            self._iteration_number = 1
            self._react_iterations = []
            self.memory.clear()

        async def _run() -> str:
            final_answer = ''
            async for output in self._run_agent(input_text):
                final_answer += output
            return final_answer

        return asyncio.run(_run())

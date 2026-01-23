from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import AsyncGenerator
from typing import Literal
import uuid

from pydantic import Field

from tinygent.agents.base_agent import TinyBaseAgent
from tinygent.agents.base_agent import TinyBaseAgentConfig
from tinygent.agents.middleware.base import AgentMiddleware
from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.memory import AbstractMemory
from tinygent.datamodels.messages import AllTinyMessages
from tinygent.datamodels.messages import TinyChatMessage
from tinygent.datamodels.messages import TinyChatMessageChunk
from tinygent.datamodels.messages import TinyHumanMessage
from tinygent.datamodels.messages import TinyReasoningMessage
from tinygent.datamodels.messages import TinySystemMessage
from tinygent.datamodels.messages import TinyToolCall
from tinygent.datamodels.tool import AbstractTool
from tinygent.factory.llm import build_llm
from tinygent.factory.memory import build_memory
from tinygent.factory.tool import build_tool
from tinygent.prompts.agents.factory.react_agent import get_prompt_template
from tinygent.prompts.agents.template.react_agent import ReActPromptTemplate
from tinygent.runtime.executors import run_async_in_executor
from tinygent.telemetry.decorators import tiny_trace
from tinygent.telemetry.otel import set_tiny_attributes
from tinygent.telemetry.otel import tiny_trace_span
from tinygent.tools.reasoning_tool import ReasoningTool
from tinygent.types.base import TinyModel
from tinygent.types.io.llm_io_chunks import TinyLLMResultChunk
from tinygent.types.io.llm_io_input import TinyLLMInput
from tinygent.utils import render_template

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = get_prompt_template()


class TinyReActAgentConfig(TinyBaseAgentConfig['TinyReActAgent']):
    """Configuration for ReAct Agent."""

    type: Literal['react'] = Field(default='react', frozen=True)

    prompt_template: ReActPromptTemplate = Field(default=_DEFAULT_PROMPT)
    max_iterations: int = Field(default=10)

    def build(self) -> TinyReActAgent:
        return TinyReActAgent(
            middleware=self.middleware,
            prompt_template=self.prompt_template,
            llm=self.llm if isinstance(self.llm, AbstractLLM) else build_llm(self.llm),
            tools=[
                tool if isinstance(tool, AbstractTool) else build_tool(tool)
                for tool in self.tools
            ],
            memory=(
                self.memory
                if isinstance(self.memory, AbstractMemory)
                else build_memory(self.memory)
            ),
            max_iterations=self.max_iterations,
        )


class TinyReActAgent(TinyBaseAgent):
    """ReAct Agent implementation."""

    def __init__(
        self,
        llm: AbstractLLM,
        memory: AbstractMemory,
        prompt_template: ReActPromptTemplate = _DEFAULT_PROMPT,
        tools: list[AbstractTool] = [],
        max_iterations: int = 10,
        middleware: Sequence[AgentMiddleware] = [],
    ) -> None:
        super().__init__(llm=llm, tools=tools, memory=memory, middleware=middleware)

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

    @tiny_trace('react_agent_reasoning')
    async def _stream_reasoning(
        self, run_id: str, task: str
    ) -> TinyChatMessage | TinyReasoningMessage:
        class TinyReasoningOutcome(TinyModel):
            type: Literal['reasoning'] = 'reasoning'
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
                *self.memory.copy_chat_messages(),
            ]
        )
        messages.add_at_beginning(
            TinySystemMessage(content=render_template(template, variables)),
        )

        result = await self.run_llm(
            run_id=run_id,
            fn=self.llm.generate_structured,
            llm_input=messages,
            output_schema=TinyReasoningOutcome,
        )

        set_tiny_attributes(
            {
                'agent.reasoning.type': result.type,
                'agent.reasoning.content': result.content,
            }
        )
        logger.debug(
            '[REASONING] - for task %s was created reasoning: %s', task, result.content
        )

        return TinyReasoningMessage(content=result.content)

    @tiny_trace('react_agent_action')
    async def _stream_action(
        self, run_id: str, reasoning: str
    ) -> AsyncGenerator[TinyLLMResultChunk, None]:
        logger.debug('[ACTION STREAM] started with reasoning: %s', reasoning)
        messages = TinyLLMInput(
            messages=[
                *self.memory.copy_chat_messages(),
            ]
        )
        messages.add_at_beginning(
            TinySystemMessage(
                content=render_template(
                    self.prompt_template.action.action,
                    {'reasoning': reasoning, 'tools': self._tools},
                )
            ),
        )

        async for chunk in self.run_llm_stream(
            run_id=run_id,
            fn=self.llm.stream_with_tools,
            llm_input=messages,
            tools=self._tools,
        ):
            yield chunk

    @tiny_trace('react_agent_fallback')
    async def _stream_fallback(
        self, run_id: str, task: str
    ) -> AsyncGenerator[str, None]:
        logger.debug('[FALLBACK] started with task: %s', task)

        messages = TinyLLMInput(
            messages=[
                *self.memory.copy_chat_messages(),
            ]
        )
        messages.add_at_beginning(
            TinySystemMessage(
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
        )

        async for chunk in self.run_llm_stream(
            run_id=run_id,
            fn=self.llm.stream_text,
            llm_input=messages,
        ):
            if isinstance(chunk, TinyLLMResultChunk) and chunk.is_message:
                assert isinstance(chunk.message, TinyChatMessageChunk)
                yield chunk.message.content

    @tiny_trace('agent_run')
    async def _run_agent(
        self, input_text: str, run_id: str
    ) -> AsyncGenerator[str, None]:
        set_tiny_attributes(
            {
                'agent.type': 'react',
                'agent.max_iterations': str(self.max_iterations),
                'agent.run_id': run_id,
                'agent.input_text': input_text,
            }
        )
        logger.debug('Running agent with task: %s', input_text)

        self._iteration_number = 1
        returned_final_answer: bool = False
        yielded_final_answer: str = ''

        self.memory.save_context(TinyHumanMessage(content=input_text))

        while not returned_final_answer and (
            self._iteration_number <= self.max_iterations
        ):
            with tiny_trace_span(
                'react_agent_single_iteration', iteration=self._iteration_number
            ):
                logger.debug('--- ITERATION %d ---', self._iteration_number)

                try:
                    reasoning_result = await self._stream_reasoning(
                        run_id=run_id, task=input_text
                    )
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

                        yield reasoning_result.content

                    else:
                        logger.debug(
                            '[%d. ITERATION - Streaming Action]', self._iteration_number
                        )

                        tool_calls: list[TinyToolCall] = []
                        async for msg in self._stream_action(
                            run_id=run_id, reasoning=reasoning_result.content
                        ):
                            if msg.is_message and isinstance(
                                msg.message, TinyChatMessageChunk
                            ):
                                returned_final_answer = True
                                yielded_final_answer += msg.message.content

                                yield msg.message.content

                            elif msg.is_tool_call and isinstance(
                                msg.full_tool_call, TinyToolCall
                            ):
                                full_tc = msg.full_tool_call
                                called_tool = self.get_tool(full_tc.tool_name)
                                if called_tool:
                                    tool_result = await self.run_tool(
                                        run_id=run_id, tool=called_tool, call=full_tc
                                    )

                                    self.memory.save_context(full_tc)
                                    self.memory.save_context(tool_result)

                                    tool_calls.append(full_tc)

                                    if isinstance(called_tool, ReasoningTool):
                                        reasoning = full_tc.arguments.get(
                                            'reasoning', ''
                                        )
                                        logger.debug(
                                            '[%d. ITERATION - Tool Reasoning]: %s',
                                            self._iteration_number,
                                            reasoning,
                                        )
                                        await self.on_tool_reasoning(
                                            run_id=run_id, reasoning=reasoning
                                        )
                                else:
                                    logger.error(
                                        'Tool %s not found. Skipping tool call.',
                                        full_tc.tool_name,
                                    )

                                logger.debug(
                                    '[%s. ITERATION - Tool Call]: %s(%s) = %s',
                                    self._iteration_number,
                                    full_tc.tool_name,
                                    full_tc.arguments,
                                    full_tc.result,
                                )

                        if yielded_final_answer:
                            self.memory.save_context(
                                TinyChatMessage(content=yielded_final_answer)
                            )

                        self._react_iterations.append(
                            self.TinyReactIteration(
                                iteration_number=self._iteration_number,
                                tool_calls=tool_calls,
                                reasoning=reasoning_result.content,
                            )
                        )
                except Exception as e:
                    logger.warning('Error happen during main react loop %s', e)
                    await self.on_error(run_id=run_id, e=e)
                    raise e
                finally:
                    self._iteration_number += 1

        if not returned_final_answer:
            logger.warning(
                'Max iterations reached without final answer. Using fallback.'
                'Returning fallback answer.'
            )

            yielded_fallback = False
            final_yielded_answer = ''

            async for fallback_chunk in self._stream_fallback(
                run_id=run_id, task=input_text
            ):
                yielded_fallback = True
                final_yielded_answer += fallback_chunk

                yield fallback_chunk

            if not yielded_fallback:
                final_yielded_answer = 'I have completed my reasoning and tool usage but did not arrive at a final answer.'
                yield final_yielded_answer

            self.memory.save_context(TinyChatMessage(content=final_yielded_answer))

    def reset(self) -> None:
        super().reset()

        logger.debug('[AGENT RESET]')

        self._iteration_number = 1
        self._react_iterations = []

    def setup(self, reset: bool, history: list[AllTinyMessages] | None) -> None:
        if reset:
            self.reset()

        if history:
            self.memory.save_multiple_context(history)

    def run(
        self,
        input_text: str,
        *,
        run_id: str | None = None,
        reset: bool = True,
        history: list[AllTinyMessages] | None = None,
    ) -> str:
        logger.debug('[USER INPUT] %s', input_text)

        run_id = run_id or str(uuid.uuid4())
        self.setup(reset=reset, history=history)

        async def _run() -> str:
            final_answer = ''
            async for output in self._run_agent(run_id=run_id, input_text=input_text):
                final_answer += output

            await self.on_answer(run_id=run_id, answer=final_answer)
            return final_answer

        return run_async_in_executor(_run)

    def run_stream(
        self,
        input_text: str,
        *,
        run_id: str | None = None,
        reset: bool = True,
        history: list[AllTinyMessages] | None = None,
    ) -> AsyncGenerator[str, None]:
        logger.debug('[USER INPUT] %s', input_text)

        run_id = run_id or str(uuid.uuid4())
        self.setup(reset=reset, history=history)

        async def _generator():
            idx = 0
            async for res in self._run_agent(run_id=run_id, input_text=input_text):
                await self.on_answer_chunk(run_id=run_id, chunk=res, idx=str(idx))
                idx += 1
                yield res

        return _generator()

    def __str__(self) -> str:
        from io import StringIO
        import textwrap

        buf = StringIO()

        extra = []
        extra.append('Type: ReAct')
        extra.append(f'Max Iterations: {self.max_iterations}')

        extra_block = '\n'.join(extra)
        extra_block = textwrap.indent(extra_block, '\t')

        buf.write(super().__str__())
        buf.write(f'{extra_block}\n')

        return buf.getvalue()

from __future__ import annotations

from collections.abc import AsyncGenerator
from collections.abc import Sequence
import logging
from typing import Any
from typing import Literal
import uuid

from pydantic import Field

from tinygent.agents.base_agent import TinyBaseAgent
from tinygent.agents.base_agent import TinyBaseAgentConfig
from tinygent.core.datamodels.llm import AbstractLLM
from tinygent.core.datamodels.memory import AbstractMemory
from tinygent.core.datamodels.messages import AllTinyMessages
from tinygent.core.datamodels.messages import TinyChatMessage
from tinygent.core.datamodels.messages import TinyChatMessageChunk
from tinygent.core.datamodels.messages import TinyHumanMessage
from tinygent.core.datamodels.messages import TinyPlanMessage
from tinygent.core.datamodels.messages import TinyReasoningMessage
from tinygent.core.datamodels.messages import TinySystemMessage
from tinygent.core.datamodels.messages import TinyToolCall
from tinygent.core.datamodels.middleware import AbstractMiddleware
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.runtime.executors import run_async_in_executor
from tinygent.core.telemetry.decorators import tiny_trace
from tinygent.core.telemetry.otel import set_tiny_attributes
from tinygent.core.telemetry.otel import tiny_trace_span
from tinygent.core.types.base import TinyModel
from tinygent.core.types.io.llm_io_chunks import TinyLLMResultChunk
from tinygent.core.types.io.llm_io_input import TinyLLMInput
from tinygent.prompts.multistep import MultiStepPromptTemplate
from tinygent.prompts.multistep import get_prompt_template
from tinygent.tools.reasoning_tool import ReasoningTool
from tinygent.utils import render_template

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = get_prompt_template()


class TinyMultiStepAgentConfig(TinyBaseAgentConfig['TinyMultiStepAgent']):
    """Configuration for the TinyMultiStepAgent."""

    type: Literal['multistep'] = Field(default='multistep', frozen=True)

    prompt_template: MultiStepPromptTemplate = Field(default=_DEFAULT_PROMPT)
    max_iterations: int = Field(default=15)
    plan_interval: int = Field(default=5)

    def build(self) -> TinyMultiStepAgent:
        return TinyMultiStepAgent(
            middleware=self.build_middleware_list(),
            llm=self.build_llm_instance(),
            tools=self.build_tools_list(),
            memory=self.build_memory_instance(),
            prompt_template=self.prompt_template,
            max_iterations=self.max_iterations,
            plan_interval=self.plan_interval,
        )


class TinyMultiStepAgent(TinyBaseAgent):
    """Multi-Step planning agent with dynamic replanning.

    This agent creates a multi-step plan to solve complex tasks and executes actions
    based on that plan. It periodically updates its plan based on progress and new
    information gathered from tool executions.

    The agent alternates between planning phases (creating/updating the plan with
    reasoning) and action phases (executing tools). Plans are refreshed at configurable
    intervals to adapt to new information. This makes it suitable for complex tasks
    requiring strategic planning and adaptation.

    Middleware Hooks Activated:
    - before_llm_call / after_llm_call - For LLM calls
    - before_tool_call / after_tool_call - For tool executions
    - on_plan - When creating initial or updated plan
    - on_reasoning - For agent reasoning steps
    - on_tool_reasoning - When reasoning tools generate reasoning
    - on_answer / on_answer_chunk - For final answers
    - on_error - On any error

    Args:
        llm: Language model for planning and actions
        memory: Memory system for maintaining conversation history
        prompt_template: Template for multi-step prompts (default provided)
        tools: List of tools available to the agent
        max_iterations: Maximum number of action iterations (default: 15)
        plan_interval: Number of iterations between plan updates (default: 5)
        middleware: List of middleware to apply during execution
    """

    def __init__(
        self,
        llm: AbstractLLM,
        memory: AbstractMemory,
        prompt_template: MultiStepPromptTemplate = _DEFAULT_PROMPT,
        tools: list[AbstractTool] = [],
        max_iterations: int = 15,
        plan_interval: int = 5,
        middleware: Sequence[AbstractMiddleware] = [],
    ) -> None:
        super().__init__(llm=llm, tools=tools, memory=memory, middleware=middleware)

        self._iteration_number: int = 1
        self._planned_steps: list[TinyPlanMessage] = []
        self._tool_calls: list[TinyToolCall] = []

        self.max_iterations = max_iterations
        self.plan_interval = plan_interval

        self.acter_prompt = prompt_template.acter
        self.plan_prompt = prompt_template.plan
        self.fallback_prompt = prompt_template.fallback

    @tiny_trace('multi_step_agent_steps_creation')
    async def _stream_steps(
        self, run_id: str, task: str
    ) -> AsyncGenerator[TinyPlanMessage | TinyReasoningMessage]:
        class TinyReasonedSteps(TinyModel):
            planned_steps: list[str]
            reasoning: str

        variables: dict[str, Any]

        # Initial plan
        if self._iteration_number == 1:
            template = self.plan_prompt.init_plan
            variables = {'task': task, 'tools': self.tools}
        else:
            template = self.plan_prompt.update_plan
            variables = {
                'task': task,
                'tools': self.tools,
                'history': self.memory.load_variables(),
                'steps': self._planned_steps,
                'remaining_steps': self.max_iterations - self._iteration_number + 1,
            }

        messages = TinyLLMInput(
            messages=[
                *self.memory.copy_chat_messages(),
            ]
        )
        messages.add_at_beginning(
            TinySystemMessage(
                content=render_template(
                    template,
                    variables,
                )
            ),
        )

        result = await self.run_llm(
            run_id=run_id,
            fn=self.llm.generate_structured,
            llm_input=messages,
            output_schema=TinyReasonedSteps,
        )

        yield TinyReasoningMessage(content=result.reasoning)
        for step in result.planned_steps:
            yield TinyPlanMessage(content=step)

        set_tiny_attributes(
            {
                'agent.planner.planned_steps': str(result.planned_steps),
                'agent.planner.num_planned_steps': str(len(result.planned_steps)),
                'agent.planner.reasoning': result.reasoning,
            }
        )

    @tiny_trace('multi_step_agent_action')
    async def _stream_action(
        self, run_id: str, task: str
    ) -> AsyncGenerator[TinyLLMResultChunk]:
        messages = TinyLLMInput(
            messages=[
                *self.memory.copy_chat_messages(),
                TinyHumanMessage(
                    content=render_template(
                        self.acter_prompt.final_answer,
                        {
                            'task': task,
                            'tools': self.tools,
                            'tool_calls': self._tool_calls,
                            'history': self.memory.load_variables(),
                            'steps': self._planned_steps,
                        },
                    )
                ),
            ]
        )
        messages.add_at_beginning(
            TinySystemMessage(content=self.acter_prompt.system),
        )

        async for chunk in self.run_llm_stream(
            run_id=run_id,
            fn=self.llm.stream_with_tools,
            llm_input=messages,
            tools=self._tools,
        ):
            yield chunk

    @tiny_trace('multi_step_agent_fallback')
    async def _stream_fallback_answer(
        self, run_id: str, task: str
    ) -> AsyncGenerator[TinyChatMessageChunk]:
        messages = TinyLLMInput(
            messages=[
                *self.memory.copy_chat_messages(),
            ]
        )
        messages.add_at_beginning(
            TinyHumanMessage(
                content=render_template(
                    self.fallback_prompt.fallback_answer,
                    {
                        'task': task,
                        'history': self.memory.load_variables(),
                        'steps': self._planned_steps,
                    },
                )
            ),
        )

        async for chunk in self.run_llm_stream(
            run_id=run_id, fn=self.llm.stream_text, llm_input=messages
        ):
            if chunk.is_message and isinstance(chunk.message, TinyChatMessageChunk):
                yield chunk.message

    @tiny_trace('agent_run')
    async def _run_agent(self, input_text: str, run_id: str) -> AsyncGenerator[str]:
        set_tiny_attributes(
            {
                'agent.type': 'multistep',
                'agent.max_iterations': str(self.max_iterations),
                'agent.plan_interval': str(self.plan_interval),
                'agent.run_id': run_id,
                'agent.input_text': input_text,
            }
        )

        logger.debug('[%s] Running agent with input %s', run_id, input_text)

        self._iteration_number = 1
        returned_final_answer: bool = False
        yielded_final_answer: str = ''

        self.memory.save_context(TinyHumanMessage(content=input_text))

        while not returned_final_answer and (
            self._iteration_number <= self.max_iterations
        ):
            with tiny_trace_span(
                'multi_step_agent_single_iteration', iteration=self._iteration_number
            ):
                logger.debug('--- ITERATION %d ---', self._iteration_number)

                if self._iteration_number == 1 or (
                    (self._iteration_number - 1) % self.plan_interval == 0
                ):
                    # Create new plan
                    plan_generator = self._stream_steps(run_id=run_id, task=input_text)
                    self._planned_steps = []

                    async for planner_msg in plan_generator:
                        if isinstance(planner_msg, TinyPlanMessage):
                            logger.debug(
                                '[%d. ITERATION - Plan]: %s',
                                self._iteration_number,
                                planner_msg.content,
                            )
                            await self.on_plan(
                                run_id=run_id, plan=planner_msg.content, kwargs={}
                            )
                            self._planned_steps.append(planner_msg)

                        if isinstance(planner_msg, TinyReasoningMessage):
                            logger.debug(
                                '[%d. ITERATION - Reasoning]: %s',
                                self._iteration_number,
                                planner_msg.content,
                            )
                            await self.on_reasoning(
                                run_id=run_id, reasoning=planner_msg.content, kwargs={}
                            )
                        self.memory.save_context(planner_msg)

                try:
                    # Execute action
                    async for msg in self._stream_action(run_id=run_id, task=input_text):
                        if msg.is_message and isinstance(
                            msg.message, TinyChatMessageChunk
                        ):
                            returned_final_answer = True
                            yielded_final_answer += msg.message.content

                            yield msg.message.content

                        elif msg.is_tool_call and isinstance(
                            msg.full_tool_call, TinyToolCall
                        ):
                            tool_call: TinyToolCall = msg.full_tool_call
                            called_tool = self.get_tool(tool_call.tool_name)

                            self.memory.save_context(tool_call)
                            if called_tool:
                                self.memory.save_context(
                                    await self.run_tool(
                                        run_id=run_id, tool=called_tool, call=tool_call
                                    )
                                )
                                self._tool_calls.append(tool_call)
                            else:
                                logger.error(
                                    'Tool %s not found. Skipping tool call.',
                                    tool_call.tool_name,
                                )

                            if isinstance(called_tool, ReasoningTool):
                                reasoning = tool_call.arguments.get('reasoning', '')
                                logger.debug(
                                    '[%d. ITERATION - Tool Reasoning]: %s',
                                    self._iteration_number,
                                    reasoning,
                                )
                                await self.on_tool_reasoning(
                                    run_id=run_id, reasoning=reasoning, kwargs={}
                                )

                            logger.debug(
                                '[%s. ITERATION - Tool Call]: %s(%s) = %s',
                                self._iteration_number,
                                tool_call.tool_name,
                                tool_call.arguments,
                                tool_call.result,
                            )

                    if returned_final_answer:
                        if yielded_final_answer:
                            self.memory.save_context(
                                TinyChatMessage(content=yielded_final_answer)
                            )
                        break
                except Exception as e:
                    await self.on_error(run_id=run_id, e=e, kwargs={})
                    raise e
                finally:
                    self._iteration_number += 1

        if not returned_final_answer:
            logger.warning(
                'Max iterations reached without returning a final answer. '
                'Returning the last known answer or a default message.'
            )

            yield_fallback = False
            final_yielded_answer = ''

            logger.debug('--- FALLBACK FINAL ANSWER ---')
            async for chunk in self._stream_fallback_answer(
                run_id=run_id, task=input_text
            ):
                yield_fallback = True
                final_yielded_answer += chunk.content

                yield chunk.content

            if not yield_fallback:
                final_yielded_answer = (
                    'I am unable to provide a final answer at this time.'
                )
                yield final_yielded_answer

            self.memory.save_context(TinyChatMessage(content=final_yielded_answer))

    def reset(self) -> None:
        super().reset()

        logger.debug('[AGENT RESET]')

        self._iteration_number = 1
        self._planned_steps = []
        self._tool_calls = []

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
            final_answer: str = ''
            async for res in self._run_agent(input_text, run_id):
                final_answer += res

            await self.on_answer(run_id=run_id, answer=final_answer, kwargs={})
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
                await self.on_answer_chunk(
                    run_id=run_id, chunk=res, idx=str(idx), kwargs={}
                )
                idx += 1
                yield res

        return _generator()

    def __str__(self) -> str:
        from io import StringIO
        import textwrap

        buf = StringIO()

        extra = []
        extra.append('Type: Multi-Step Agent')
        extra.append(f'Max Iterations: {self.max_iterations}')
        extra.append(f'Plan Interval: {self.plan_interval}')

        extra_block = '\n'.join(extra)
        extra_block = textwrap.indent(extra_block, '\t')

        buf.write(super().__str__())
        buf.write(f'{extra_block}\n')

        return buf.getvalue()

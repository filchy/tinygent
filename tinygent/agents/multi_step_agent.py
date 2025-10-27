from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from collections.abc import Generator
import logging
import typing
from typing import Any
from typing import Literal

from tinygent.agents import TinyBaseAgent
from tinygent.agents import TinyBaseAgentConfig
from tinygent.cli.builder import build_llm
from tinygent.cli.builder import build_memory
from tinygent.cli.builder import build_tool
from tinygent.datamodels.llm_io_chunks import TinyLLMResultChunk
from tinygent.datamodels.llm_io_input import TinyLLMInput
from tinygent.datamodels.memory import AbstractMemory
from tinygent.datamodels.messages import TinyChatMessage
from tinygent.datamodels.messages import TinyChatMessageChunk
from tinygent.datamodels.messages import TinyHumanMessage
from tinygent.datamodels.messages import TinyPlanMessage
from tinygent.datamodels.messages import TinyReasoningMessage
from tinygent.datamodels.messages import TinySystemMessage
from tinygent.datamodels.messages import TinyToolCall
from tinygent.datamodels.prompt import TinyPromptTemplate
from tinygent.memory import BufferChatMemory
from tinygent.tools.default_tools import provide_final_answer
from tinygent.tools.reasoning_tool import ReasoningTool
from tinygent.types.base import TinyModel
from tinygent.utils import is_final_answer
from tinygent.utils import render_template

if typing.TYPE_CHECKING:
    from tinygent.datamodels.llm import AbstractLLM
    from tinygent.datamodels.tool import AbstractTool

logger = logging.getLogger(__name__)


class PlanPromptTemplate(TinyPromptTemplate):
    """Used to generate or update the plan."""

    init_plan: str
    update_plan: str

    _template_fields = {
        'init_plan': {'task', 'tools'},
        'update_plan': {'task', 'tools', 'history', 'steps', 'remaining_steps'},
    }


class ActionPromptTemplate(TinyPromptTemplate):
    """Used to generate the final answer or action."""

    system: str
    final_answer: str

    _template_fields = {
        'final_answer': {'task', 'tools', 'history', 'steps', 'tool_calls'},
    }


class FallbackAnswerPromptTemplate(TinyPromptTemplate):
    """Used to generate the final answer if maximum steps achieved."""

    fallback_answer: str

    _template_fields = {
        'fallback_answer': {'task', 'history', 'steps'},
    }


class MultiStepPromptTemplate(TinyPromptTemplate):
    """Prompt templates for the multi-step agent."""

    plan: PlanPromptTemplate
    acter: ActionPromptTemplate
    fallback: FallbackAnswerPromptTemplate


class TinyMultiStepAgentConfig(TinyBaseAgentConfig['TinyMultiStepAgent']):
    """Configuration for the TinyMultiStepAgent."""

    type: Literal['multistep'] = 'multistep'

    prompt_template: MultiStepPromptTemplate
    max_iterations: int = 15
    plan_interval: int = 5

    def build(self) -> TinyMultiStepAgent:
        return TinyMultiStepAgent(
            llm=build_llm(self.llm),
            prompt_template=self.prompt_template,
            tools=[build_tool(tool) for tool in self.tools],
            memory=build_memory(self.memory),
            max_iterations=self.max_iterations,
            plan_interval=self.plan_interval,
        )


class TinyMultiStepAgent(TinyBaseAgent):
    """Multi-Step Agent implementation."""

    def __init__(
        self,
        llm: AbstractLLM,
        prompt_template: MultiStepPromptTemplate,
        memory: AbstractMemory = BufferChatMemory(),
        tools: list[AbstractTool] = [],
        max_iterations: int = 15,
        plan_interval: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(llm=llm, tools=tools, memory=memory, **kwargs)

        self._iteration_number: int = 1
        self._planned_steps: list[TinyPlanMessage] = []
        self._tool_calls: list[TinyToolCall] = []

        __all_tools = list(tools) + [provide_final_answer]
        self._tools: list[AbstractTool] = [tool for tool in __all_tools]

        self.max_iterations = max_iterations
        self.plan_interval = plan_interval

        self.acter_prompt = prompt_template.acter
        self.plan_prompt = prompt_template.plan
        self.fallback_prompt = prompt_template.fallback

    def _stream_steps(
        self, task: str
    ) -> Generator[TinyPlanMessage | TinyReasoningMessage]:
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
                *self.memory.chat_messages,
                TinyHumanMessage(
                    content=render_template(
                        template,
                        variables,
                    )
                ),
            ]
        )

        result = self.run_llm(
            self.llm.generate_structured,
            llm_input=messages,
            output_schema=TinyReasonedSteps,
        )

        yield TinyReasoningMessage(content=result.reasoning)
        for step in result.planned_steps:
            yield TinyPlanMessage(content=step)

    async def _stream_action(self, task: str) -> AsyncGenerator[TinyLLMResultChunk]:
        messages = TinyLLMInput(
            messages=[
                *self.memory.chat_messages,
                TinySystemMessage(content=self.acter_prompt.system),
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

        async for chunk in self.run_llm_stream(
            self.llm.stream_with_tools, llm_input=messages, tools=self._tools
        ):
            yield chunk

    async def _stream_fallback_answer(
        self, task: str
    ) -> AsyncGenerator[TinyChatMessageChunk]:
        messages = TinyLLMInput(
            messages=[
                *self.memory.chat_messages,
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
            ]
        )

        async for chunk in self.run_llm_stream(self.llm.stream_text, llm_input=messages):
            if chunk.is_message and isinstance(chunk.message, TinyChatMessageChunk):
                yield chunk.message

    async def _run_agent(self, input_text: str) -> AsyncGenerator[str]:
        self._iteration_number = 1
        returned_final_answer: bool = False
        yielded_final_answer: str = ''

        self.memory.save_context(TinyHumanMessage(content=input_text))

        while not returned_final_answer and (
            self._iteration_number <= self.max_iterations
        ):
            logger.debug('--- ITERATION %d ---', self._iteration_number)

            if self._iteration_number == 1 or (
                (self._iteration_number - 1) % self.plan_interval == 0
            ):
                # Create new plan
                plan_generator = self._stream_steps(input_text)
                self._planned_steps = []

                for planner_msg in plan_generator:
                    if isinstance(planner_msg, TinyPlanMessage):
                        logger.debug(
                            '[%d. ITERATION - Plan]: %s',
                            self._iteration_number,
                            planner_msg.content,
                        )
                        self.on_plan(planner_msg.content)
                        self._planned_steps.append(planner_msg)
                    if isinstance(planner_msg, TinyReasoningMessage):
                        logger.debug(
                            '[%d. ITERATION - Reasoning]: %s',
                            self._iteration_number,
                            planner_msg.content,
                        )
                        self.on_reasoning(planner_msg.content)
                    self.memory.save_context(planner_msg)

            try:
                # Execute action
                async for msg in self._stream_action(input_text):
                    if msg.is_message and isinstance(msg.message, TinyChatMessageChunk):
                        returned_final_answer = True
                        yielded_final_answer += msg.message.content
                        yield msg.message.content

                    elif msg.is_tool_call and isinstance(
                        msg.full_tool_call, TinyToolCall
                    ):
                        tool_call: TinyToolCall = msg.full_tool_call
                        self.memory.save_context(tool_call)
                        called_tool = self.get_tool(tool_call.tool_name)
                        if called_tool:
                            self.memory.save_context(
                                self.run_tool(called_tool, tool_call)
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
                            self.on_tool_reasoning(reasoning)

                        logger.debug(
                            '[%s. ITERATION - Tool Call]: %s(%s) = %s',
                            self._iteration_number,
                            tool_call.tool_name,
                            tool_call.arguments,
                            tool_call.result,
                        )

                        if isinstance(
                            tool_call.result, TinyChatMessage
                        ) and is_final_answer(tool_call.result):
                            returned_final_answer = True

                            self.memory.save_context(tool_call.result)
                            self.on_answer(tool_call.result.content)
                            yield tool_call.result.content

                    if returned_final_answer:
                        if yielded_final_answer:
                            self.memory.save_context(
                                TinyChatMessage(content=yielded_final_answer)
                            )
                            self.on_answer(yielded_final_answer)
                        break
            except Exception as e:
                self.on_error(e)
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
            async for chunk in self._stream_fallback_answer(input_text):
                yield_fallback = True
                final_yielded_answer += chunk.content

                yield chunk.content

            if not yield_fallback:
                final_yielded_answer = (
                    'I am unable to provide a final answer at this time.'
                )
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
            self._planned_steps = []
            self._tool_calls = []
            self.memory.clear()

        async def _run() -> str:
            final_answer: str = ''
            async for res in self._run_agent(input_text):
                final_answer += res
            return final_answer

        return asyncio.run(_run())

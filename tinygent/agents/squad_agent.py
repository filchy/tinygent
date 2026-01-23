from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging
from typing import AsyncGenerator
from typing import Literal
from typing import Self
from typing import cast
import uuid

from pydantic import Field
from pydantic import model_validator

from tinygent.agents.base_agent import TinyBaseAgent
from tinygent.agents.base_agent import TinyBaseAgentConfig
from tinygent.agents.middleware.base import AgentMiddleware
from tinygent.datamodels.agent import AbstractAgent
from tinygent.datamodels.agent import AbstractAgentConfig
from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.memory import AbstractMemory
from tinygent.datamodels.messages import AllTinyMessages
from tinygent.datamodels.messages import TinyHumanMessage
from tinygent.datamodels.messages import TinySquadMemberMessage
from tinygent.datamodels.messages import TinySystemMessage
from tinygent.datamodels.tool import AbstractTool
from tinygent.factory.agent import build_agent
from tinygent.factory.llm import build_llm
from tinygent.factory.memory import build_memory
from tinygent.factory.middleware import build_middleware
from tinygent.factory.tool import build_tool
from tinygent.prompts.agents.factory.squad_agent import get_prompt_template
from tinygent.prompts.agents.template.squad_agent import SquadPromptTemplate
from tinygent.runtime.executors import run_async_in_executor
from tinygent.telemetry.decorators import tiny_trace
from tinygent.telemetry.otel import set_tiny_attributes
from tinygent.telemetry.otel import tiny_trace_span
from tinygent.types.base import TinyModel
from tinygent.types.io.llm_io_input import TinyLLMInput
from tinygent.utils.jinja_utils import render_template

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = get_prompt_template()


class ClassificationQueryResult(TinyModel):
    selected_member: str = Field(
        ..., description='The name of the selected squad member to handle the task.'
    )

    task: str = Field(..., description='The task assigned to the selected squad member.')

    reasoning: str = Field(
        ..., description='The reasoning behind the selection of the squad member.'
    )


@dataclass(frozen=True)
class AgentSquadMemberConfig:
    """Configuration for a member of the agent squad."""

    name: str
    description: str
    agent: dict | AbstractAgentConfig | str


@dataclass(frozen=True)
class AgentSquadMember:
    """A member of the agent squad."""

    name: str
    description: str
    agent: AbstractAgent

    @classmethod
    def from_config(cls, config: AgentSquadMemberConfig) -> 'AgentSquadMember':
        return cls(
            name=config.name,
            description=config.description,
            agent=build_agent(config.agent),
        )


class TinySquadAgentConfig(TinyBaseAgentConfig['TinySquadAgent']):
    """Configuration for TinySquadAgent."""

    type: Literal['squad'] = Field(default='squad')

    prompt_template: SquadPromptTemplate = Field(default=_DEFAULT_PROMPT)
    squad: list[AgentSquadMemberConfig] = Field(...)

    def build(self) -> TinySquadAgent:
        return TinySquadAgent(
            middleware=[build_middleware(m) for m in self.middleware],
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
            squad=[AgentSquadMember.from_config(agent_cfg) for agent_cfg in self.squad],
        )

    @model_validator(mode='after')
    def validate_agent(self) -> Self:
        if not self.squad or len(self.squad) == 0:
            raise ValueError('Squad agent must have at least one squad member.')

        return self


class TinySquadAgent(TinyBaseAgent):
    """Squad Agent for coordinating multiple agents to solve complex tasks."""

    def __init__(
        self,
        llm: AbstractLLM,
        memory: AbstractMemory,
        prompt_template: SquadPromptTemplate = _DEFAULT_PROMPT,
        tools: list[AbstractTool] = [],
        squad: list[AgentSquadMember] = [],
        middleware: Sequence[AgentMiddleware] = [],
    ) -> None:
        super().__init__(llm=llm, tools=tools, memory=memory, middleware=middleware)

        self._squad = [self._normalize_squad_member(member) for member in squad]

        self.prompt_template = prompt_template

    @staticmethod
    def _normalize_squad_member(member: AgentSquadMember) -> AgentSquadMember:
        def _empty(*_args, **_kwargs) -> None:
            return None

        for m in member.agent.middleware:
            m.on_answer = _empty  # type: ignore[method-assign]
            m.on_answer_chunk = _empty  # type: ignore[method-assign]

        return member

    def _get_squad_member(self, name: str) -> AgentSquadMember:
        logger.debug('Getting squad member: %s', name)
        selected_member = next(
            (member for member in self._squad if member.name == name), None
        )

        if selected_member is None:
            logger.warning('Could not get member %s', name)
            raise ValueError(f'Squad member "{name}" not found.')

        logger.debug('Squad member(%s) succesfully found', name)
        return selected_member

    @tiny_trace('classify_query')
    def _classify_query(self, run_id: str, input_text: str) -> ClassificationQueryResult:
        logger.debug('[CLASSIFY QUERY] classifying query: %s', input_text)

        _ValidMemberNames = Literal[tuple([member.name for member in self._squad])]  # type: ignore

        class _ClassificationQueryResult(TinyModel):
            selected_member: _ValidMemberNames = Field(  # type: ignore
                ...,
                description='The name of the selected squad member to handle the task.',
            )

            task: str = Field(
                ..., description='The task assigned to the selected squad member.'
            )

            reasoning: str = Field(
                ...,
                description='The reasoning behind the selection of the squad member.',
            )

        messages = TinyLLMInput(messages=[*self.memory.copy_chat_messages()])
        messages.add_at_beginning(
            TinySystemMessage(
                content=render_template(
                    self.prompt_template.classifier.prompt,
                    {
                        'task': input_text,
                        'tools': self._tools,
                        'squad_members': self._squad,
                    },
                )
            )
        )

        response = self.run_llm(
            run_id=run_id,
            fn=self.llm.generate_structured,
            llm_input=messages,
            output_schema=_ClassificationQueryResult,
        )

        set_tiny_attributes(
            {
                'agent.classifier.assigned_member': response.selected_member,
                'agent.classifier.assigned_task': response.task,
                'agent.classifier.reasoning': response.reasoning,
            }
        )
        logger.debug(
            '[CLASSIFY QUERY] query: %s selected member: %s task: %s reasoning: %s',
            input_text,
            response.selected_member,
            response.task,
            response.reasoning,
        )

        return cast(ClassificationQueryResult, response)

    @tiny_trace('agent_run')
    async def _run_agent(
        self, input_text: str, run_id: str
    ) -> AsyncGenerator[str, None]:
        set_tiny_attributes(
            {
                'agent.type': 'squad',
                'agent.run_id': run_id,
                'agent.input_text': input_text,
                'agent.squad_size': len(self._squad),
                'agent.squad_members': ','.join(
                    f'{member.name} - {member.description}' for member in self._squad
                ),
            }
        )
        logger.debug('Running agent with task: %s', input_text)

        final_answer = ''
        self.memory.save_context(TinyHumanMessage(content=input_text))

        try:
            classification_result = self._classify_query(
                run_id=run_id, input_text=input_text
            )
            selected_member = self._get_squad_member(
                classification_result.selected_member
            )

            with tiny_trace_span('selected_squad_member'):
                async for msg in selected_member.agent.run_stream(
                    input_text=classification_result.task,
                    run_id=run_id,
                ):
                    final_answer += msg
                    yield msg

                self.memory.save_context(
                    TinySquadMemberMessage(
                        member_name=selected_member.name,
                        task=classification_result.task,
                        result=final_answer,
                    )
                )
                self.memory.save_context(TinyHumanMessage(content=final_answer))
        except Exception as e:
            self.on_error(run_id=run_id, e=e)
            raise e

    def reset(self) -> None:
        logger.debug('[AGENT RESET]')
        super().reset()

        self.memory.clear()

        for member in self._squad:
            member.agent.reset()

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

            self.on_answer(run_id=run_id, answer=final_answer)
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
                self.on_answer_chunk(run_id=run_id, chunk=res, idx=str(idx))
                idx += 1
                yield res

        return _generator()

    def __str__(self) -> str:
        from io import StringIO
        import textwrap

        buf = StringIO()

        extra = []
        extra.append(f'Squad Members ({len(self._squad)}):')
        extra.extend(
            textwrap.indent(
                f'- {member.name}: {member.description}'
                f'{textwrap.indent(str(member.agent), "\t")}',
                '\t',
            )
            for member in self._squad
        )

        extra_block = '\n'.join(extra)
        extra_block = textwrap.indent(extra_block, '\t')

        buf.write(super().__str__())
        buf.write(f'{extra_block}\n')

        return buf.getvalue()

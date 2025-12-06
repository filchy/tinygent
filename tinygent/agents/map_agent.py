from __future__ import annotations

import asyncio
import logging
import typing
from typing import AsyncGenerator
from typing import Literal
import uuid

from tinygent.agents.base_agent import TinyBaseAgent
from tinygent.agents.base_agent import TinyBaseAgentConfig
from tinygent.cli.builder import build_llm
from tinygent.cli.builder import build_memory
from tinygent.cli.builder import build_tool
from tinygent.datamodels.llm_io_input import TinyLLMInput
from tinygent.datamodels.memory import AbstractMemory
from tinygent.datamodels.messages import AllTinyMessages
from tinygent.datamodels.messages import TinyChatMessage
from tinygent.datamodels.messages import TinyHumanMessage
from tinygent.datamodels.messages import TinySystemMessage
from tinygent.runtime.executors import run_async_in_executor
from tinygent.telemetry.decorators import tiny_trace
from tinygent.telemetry.otel import set_tiny_attributes
from tinygent.types.base import TinyModel
from tinygent.types.prompt_template import TinyPromptTemplate
from tinygent.utils.jinja_utils import render_template

if typing.TYPE_CHECKING:
    from tinygent.datamodels.llm import AbstractLLM
    from tinygent.datamodels.tool import AbstractTool

logger = logging.getLogger(__name__)


class TinyMAPActionProposal(TinyModel):
    """Actor proposal plan."""

    question: str

    answer: str

    @property
    def sum(self) -> str:
        return 'Sub-question: %s \nAnswer: %s' % (self.question, self.answer)


class TinyMAPState(TinyModel):
    """Predictors prediction."""

    is_valid: bool

    next_state: str

    reason: str

    metadata: str


class TinyMAPSearchResult(TinyModel):
    """Result of the 'search' component containing 'next state', 'proposed action' and its eval score."""

    next_state: TinyMAPState

    action: TinyMAPActionProposal

    eval_score: TinyMAPEvaluatorResult


class TinyMAPEvaluatorResult(TinyModel):
    """Evaluator result."""

    score: int


class TinyMAPOrchestratorResult(TinyModel):
    """Orchestrator result."""

    fully_satisfies: bool


class TinyMAPMonitorValidity(TinyModel):
    """Monitor validity result."""

    orig_question: str

    orig_answer: str

    is_valid: bool

    feedback: str

    @property
    def validation(self) -> str:
        return 'Valid response' if self.is_valid else 'NOT-Valid response'


class OrchestratorPromptTemplate(TinyPromptTemplate, TinyPromptTemplate.UserSystem):
    """Used to define orchestrator prompt template."""

    _template_fields = {'user': {'question', 'answer'}}


class EvaluatorPromptTemplate(TinyPromptTemplate, TinyPromptTemplate.UserSystem):
    """Used to define evaluator prompt template."""

    _template_fields = {'user': {'state', 'subgoal'}}


class MonitorPrompTemplate(TinyPromptTemplate):
    """Used to define monitor prompt template."""

    init: TinyPromptTemplate.UserSystem

    continuos: TinyPromptTemplate.UserSystem

    _template_fields = {
        'init.user': {'question', 'answer'},
        'continuos.user': {'question', 'answer', 'previous_questions'},
    }


class ActorPromptTemplate(TinyPromptTemplate):
    """Used to define actor prompt template."""

    init: TinyPromptTemplate.UserSystem

    init_fixer: TinyPromptTemplate.UserSystem

    continuos: TinyPromptTemplate.UserSystem

    continuos_fixer: TinyPromptTemplate.UserSystem

    evaluator: EvaluatorPromptTemplate

    _template_fields = {
        'init.user': {'question'},
        'init_fixer.user': {'question', 'validation'},
        'continuos.user': {'question', 'previous_questions'},
        'continuos_fixer.user': {'question', 'validation'},
        'evaluator.user': {'state', 'subgoal'},
    }


class ActionProposalPromptTemplate(TinyModel):
    """Used to define action proposal module prompt template."""

    actor: ActorPromptTemplate

    monitor: MonitorPrompTemplate


class TaskDecomposerPromptTemplate(TinyPromptTemplate, TinyPromptTemplate.UserSystem):
    """Used to define task decomposer prompt template."""

    _template_fields = {'user': {'question', 'max_subquestions'}}


class PredictorPromptTemplate(TinyPromptTemplate, TinyPromptTemplate.UserSystem):
    """Used to define predictor prompt template."""

    _template_fields = {'user': {'state', 'proposed_action'}}


class MapPromptTemplate(TinyModel):
    """Prompt template for MAP Agent."""

    task_decomposer: TaskDecomposerPromptTemplate

    action_proposal: ActionProposalPromptTemplate

    predictor: PredictorPromptTemplate

    orchestrator: OrchestratorPromptTemplate


class TinyMAPAgentConfig(TinyBaseAgentConfig['TinyMAPAgent']):
    """Configuration for the TinyMAPAgent."""

    type: Literal['map'] = 'map'

    prompt_template: MapPromptTemplate

    max_plan_length: int

    max_branches_per_layer: int

    max_layer_depth: int

    max_recurrsion: int = 5

    def build(self) -> TinyMAPAgent:
        return TinyMAPAgent(
            prompt_template=self.prompt_template,
            llm=build_llm(self.llm),
            memory=build_memory(self.memory),
            tools=[build_tool(tool) for tool in self.tools],
            max_plan_length=self.max_plan_length,
            max_branches_per_layer=self.max_branches_per_layer,
            max_layer_depth=self.max_layer_depth,
            max_recurrsion=self.max_recurrsion,
        )


class TinyMAPAgent(TinyBaseAgent):
    """MAP Agent implementation."""

    def __init__(
        self,
        llm: AbstractLLM,
        prompt_template: MapPromptTemplate,
        memory: AbstractMemory,
        max_plan_length: int,
        max_branches_per_layer: int,
        max_layer_depth: int,
        max_recurrsion: int = 5,
        tools: list[AbstractTool] = [],
        **kwargs,
    ) -> None:
        super().__init__(llm=llm, tools=tools, memory=memory, **kwargs)

        self.max_plan_length = max_plan_length
        self.max_branches_per_layer = max_branches_per_layer
        self.max_recurrsion = max_recurrsion
        self.max_layer_depth = max_layer_depth
        self.prompt_template = prompt_template

    @tiny_trace('map_agent_task_decomposer')
    def _task_decomposer(self, run_id: str, input_txt: str) -> list[str]:
        class DecomposedTask(TinyModel):
            class subgoal(TinyModel):
                index: int
                question: str

            subgoals: list[subgoal]

        messages = TinyLLMInput(messages=[*self.memory.copy_chat_messages()])
        messages.add_at_end(
            TinyHumanMessage(
                content=render_template(
                    self.prompt_template.task_decomposer.user,
                    {'question': input_txt, 'max_subquestions': self.max_plan_length},
                )
            )
        )
        messages.add_before_last(
            TinySystemMessage(content=self.prompt_template.task_decomposer.system)
        )

        result = self.run_llm(
            run_id=run_id,
            fn=self.llm.generate_structured,
            llm_input=messages,
            output_schema=DecomposedTask,
        )

        all_subgoals = [f'{sq.index}. {sq.question}' for sq in result.subgoals]

        set_tiny_attributes(
            {
                'agent.map.task_decomposer.subgoals': '\n'.join(all_subgoals),
                'agent.map.task_decomposer.num_subgoals': len(all_subgoals),
            }
        )
        return all_subgoals

    @tiny_trace('map_agent_actor')
    def _actor(
        self,
        run_id: str,
        subgoal: str,
        prev_proposals: list[TinyMAPActionProposal],
        feedback: list[str],
    ) -> str:
        prompt_templ = (
            self.prompt_template.action_proposal.actor.continuos
            if prev_proposals
            else self.prompt_template.action_proposal.actor.init
        )

        formatted_proposals = [p.sum for p in prev_proposals]
        messages = TinyLLMInput(messages=[*self.memory.copy_chat_messages()])
        messages.add_at_end(
            TinyHumanMessage(
                content=render_template(
                    prompt_templ.user,
                    {
                        'question': subgoal,
                        'previous_questions': '\n'.join(formatted_proposals),
                    },
                )
            )
        )
        messages.add_before_last(TinySystemMessage(content=prompt_templ.system))
        if feedback:
            fix_prompt_templ = (
                self.prompt_template.action_proposal.actor.continuos_fixer
                if prev_proposals
                else self.prompt_template.action_proposal.actor.init_fixer
            )

            messages.add_at_end(
                TinyChatMessage(
                    content=subgoal,
                )
            )
            messages.add_at_end(
                TinyHumanMessage(
                    content=render_template(
                        fix_prompt_templ.user,
                        {'question': subgoal, 'validation': '\n'.join(feedback)},
                    )
                )
            )

        result = self.run_llm(
            run_id=run_id, fn=self.llm.generate_text, llm_input=messages
        )

        subanswer = ' '.join(
            (gen.message.content or '').strip()
            for group in result.generations
            for gen in group
        )

        set_tiny_attributes(
            {
                'agent.map.search.action_proposal.actor.subgoal': subgoal,
                'agent.map.search.action_proposal.actor.subanswer': subanswer,
                'agent.map.search.action_proposal.actor.prev_subgoals': '\n'.join(
                    formatted_proposals
                ),
                'agent.map.search.action_proposal.actor.is_repairing': bool(feedback),
            }
        )
        return subanswer

    @tiny_trace('map_agent_monitor')
    def _monitor(
        self,
        run_id: str,
        current_proposal: TinyMAPActionProposal,
        prev_proposals: list[TinyMAPActionProposal] = [],
    ) -> TinyMAPMonitorValidity:
        class _MonitorResult(TinyModel):
            is_valid: bool

            feedback: str

        prompt_templ = (
            self.prompt_template.action_proposal.monitor.continuos
            if prev_proposals
            else self.prompt_template.action_proposal.monitor.init
        )

        formatted_proposals = [p.sum for p in prev_proposals]
        messages = TinyLLMInput()
        messages.add_at_end(
            TinyHumanMessage(
                content=render_template(
                    prompt_templ.user,
                    {
                        'question': current_proposal.question,
                        'answer': current_proposal.answer,
                        'previous_questions': '\n'.join(formatted_proposals),
                    },
                )
            )
        )
        messages.add_before_last(TinySystemMessage(content=prompt_templ.system))

        result = self.run_llm(
            run_id=run_id,
            fn=self.llm.generate_structured,
            llm_input=messages,
            output_schema=_MonitorResult,
        )

        set_tiny_attributes(
            {
                'agent.map.search.action_proposal.monitor.current_question': current_proposal.question,
                'agent.map.search.action_proposal.monitor.current_answer': current_proposal.answer,
                'agent.map.search.action_proposal.monitor.previous_proposals': '\n'.join(
                    formatted_proposals
                ),
                'agent.map.search.action_proposal.monitor.result.is_valid': result.is_valid,
                'agent.map.search.action_proposal.monitor.result.feedback': result.feedback,
            }
        )

        return TinyMAPMonitorValidity(
            orig_question=current_proposal.question,
            orig_answer=current_proposal.answer,
            is_valid=result.is_valid,
            feedback=result.feedback,
        )

    @tiny_trace('map_agent_single_action_proposal')
    async def _single_action_proposal(
        self, run_id: str, subgoal: str, prev_actions: list[TinyMAPActionProposal]
    ) -> TinyMAPActionProposal:
        num_tries = 0
        validity = False

        feedback: list[str] = []
        all_proposals: list[TinyMAPActionProposal] = []

        while not validity and num_tries < self.max_recurrsion:
            subanswer = self._actor(run_id, subgoal, prev_actions, feedback)

            proposal = TinyMAPActionProposal(question=subgoal, answer=subanswer)

            monitor_validity = self._monitor(run_id, proposal, prev_actions)
            validity = monitor_validity.is_valid

            all_proposals.append(proposal)
            feedback.append(monitor_validity.feedback)

            num_tries += 1

        if not (proposal := all_proposals[-1]):
            raise RuntimeError(
                'Failed to generate action proposal in reccursion limit (%d)',
                self.max_recurrsion,
            )

        set_tiny_attributes(
            {
                'agent.map.search.action_proposal.single_action_proposal.num_tries': num_tries,
                'agent.map.search.action_proposal.single_action_proposal.all_proposals': '\n'.join(
                    [p.sum for p in all_proposals]
                ),
                'agent.map.search.action_proposal.single_action_proposal.all_feedback': '\n'.join(
                    feedback
                ),
            }
        )

        return proposal

    @tiny_trace('map_agent.map.search.action_proposal')
    async def _action_proposal(
        self, run_id: str, subgoal: str, prev_actions: list[TinyMAPActionProposal]
    ) -> list[TinyMAPActionProposal]:
        tasks = [
            asyncio.create_task(
                self._single_action_proposal(run_id, subgoal, prev_actions)
            )
            for _ in range(self.max_branches_per_layer)
        ]

        proposals: list[TinyMAPActionProposal] = await asyncio.gather(*tasks)

        formatted_proposals = '\n'.join(
            [f'{i} - {p.sum}' for i, p in enumerate(proposals)]
        )

        set_tiny_attributes(
            {
                'agent.map.search.action_proposal.num_proposals': len(proposals),
                'agent.map.search.action_proposal.proposals': formatted_proposals,
            }
        )
        return proposals

    @tiny_trace('map_agent_predictor')
    def _predictor(
        self, run_id: str, state: TinyMAPState, action: TinyMAPActionProposal
    ) -> TinyMAPState:
        messages = TinyLLMInput()
        messages.add_at_end(
            TinyHumanMessage(
                content=render_template(
                    self.prompt_template.predictor.user,
                    {'state': state.next_state, 'proposed_action': action.sum},
                )
            )
        )
        messages.add_before_last(
            TinySystemMessage(content=self.prompt_template.predictor.system)
        )

        result = self.run_llm(
            run_id=run_id,
            fn=self.llm.generate_structured,
            llm_input=messages,
            output_schema=TinyMAPState,
        )

        set_tiny_attributes(
            {
                'agent.map.search.predictor.is_valid': result.is_valid,
                'agent.map.search.predictor.next_state': result.next_state,
                'agent.map.search.predictor.reason': result.reason,
                'agent.map.search.predictor.metadata': result.metadata,
            }
        )
        return result

    @tiny_trace('map_agent_search')
    async def _search(
        self,
        run_id: str,
        depth: int,  # l
        state: TinyMAPState,  # x
        subgoal: str,  # y
    ) -> TinyMAPSearchResult:
        eval_values: list[TinyMAPEvaluatorResult] = []
        next_states: list[TinyMAPState] = []
        actions: list[TinyMAPActionProposal] = []

        proposed_actions = await self._action_proposal(run_id, subgoal, [])

        for action in proposed_actions:
            pred_state = self._predictor(run_id, state, action)

            orch_res = self._orchestrator(run_id, pred_state, subgoal)

            if depth < self.max_layer_depth and not orch_res.fully_satisfies:
                child_res = await self._search(run_id, depth + 1, pred_state, subgoal)

                l_next_state = child_res.next_state
                l_eval_score = child_res.eval_score
                l_action = child_res.action
            else:
                l_next_state = pred_state
                l_eval_score = self._evaluator(run_id, pred_state, subgoal)
                l_action = action

            eval_values.append(l_eval_score)
            next_states.append(l_next_state)
            actions.append(l_action)

        best_i = eval_values.index(max(eval_values, key=lambda x: x.score))

        best_state = next_states[best_i]
        best_action = actions[best_i]
        best_eval_score = eval_values[best_i]

        set_tiny_attributes(
            {
                'agent.map.search.best_state': best_state.next_state,
                'agent.map.search.best_action': best_action.sum,
                'agent.map.search.best_eval_score': best_eval_score.score,
            }
        )

        return TinyMAPSearchResult(
            next_state=best_state,
            action=best_action,
            eval_score=best_eval_score,
        )

    @tiny_trace('map_agent_evaluator')
    def _evaluator(
        self, run_id: str, state: TinyMAPState, subgoal: str
    ) -> TinyMAPEvaluatorResult:
        messages = TinyLLMInput()
        messages.add_at_end(
            TinyHumanMessage(
                content=render_template(
                    self.prompt_template.action_proposal.actor.evaluator.user,
                    {'state': state.next_state, 'subgoal': subgoal},
                )
            )
        )
        messages.add_before_last(
            TinySystemMessage(
                content=self.prompt_template.action_proposal.actor.evaluator.system,
            )
        )

        result = self.run_llm(
            run_id=run_id,
            fn=self.llm.generate_structured,
            llm_input=messages,
            output_schema=TinyMAPEvaluatorResult,
        )

        set_tiny_attributes(
            {
                'agent.map.search.evaluator.state': state.next_state,
                'agent.map.search.evaluator.subgoal': subgoal,
                'agent.map.search.evaluator.score': result.score,
            }
        )
        return result

    @tiny_trace('map_agent_orchestrator')
    def _orchestrator(
        self, run_id: str, state: TinyMAPState, subgoal: str
    ) -> TinyMAPOrchestratorResult:
        messages = TinyLLMInput()
        messages.add_at_end(
            TinyHumanMessage(
                content=render_template(
                    self.prompt_template.orchestrator.user,
                    {'question': subgoal, 'answer': state.next_state},
                )
            )
        )
        messages.add_before_last(
            TinySystemMessage(content=self.prompt_template.orchestrator.system)
        )

        result = self.run_llm(
            run_id=run_id,
            fn=self.llm.generate_structured,
            llm_input=messages,
            output_schema=TinyMAPOrchestratorResult,
        )

        set_tiny_attributes(
            {
                'agent.orchestrator.subgoal': subgoal,
                'agent.orchestrator.next_state': state.next_state,
            }
        )
        return result

    @tiny_trace('map_agent_map')
    async def _map(self, run_id: str, question: str) -> list[TinyMAPActionProposal]:
        subgoals = self._task_decomposer(run_id, question)
        subgoals.append(
            question
        )  # INFO: Last and final subgoal is original user question

        final_plan: list[TinyMAPActionProposal] = []

        for subgoal in subgoals:
            current_state = TinyMAPState(
                is_valid=True,
                next_state=f'Initial problem context: {question}',
                reason='initial state',
                metadata='',
            )

            validity = self._orchestrator(run_id, current_state, subgoal)

            while (
                not validity.fully_satisfies and len(final_plan) < self.max_plan_length
            ):
                search_res = await self._search(run_id, 0, current_state, subgoal)

                final_plan.append(search_res.action)
                current_state = search_res.next_state
                validity = self._orchestrator(run_id, current_state, subgoal)

                self.on_plan(run_id=run_id, plan=search_res.action.sum)

        set_tiny_attributes(
            {'agent.map.final_plan': '\n'.join([p.sum for p in final_plan])}
        )
        return final_plan

    @tiny_trace('agent_run')
    async def _run_agent(self, input_text: str, run_id: str) -> str:
        set_tiny_attributes(
            {
                'agent.type': 'map',
                'agent.run_id': run_id,
                'agent.input_text': input_text,
            }
        )

        self.memory.save_context(TinyHumanMessage(content=input_text))

        try:
            final_plan = await self._map(run_id, input_text)

            return '\n\n'.join([p.sum for p in final_plan])
        except Exception as e:
            self.on_error(run_id=run_id, e=e)
            raise e

    def reset(self) -> None:
        super().reset()

        logger.debug('[AGENT RESET]')

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
            plan = await self._run_agent(run_id=run_id, input_text=input_text)

            self.on_answer(run_id=run_id, answer=plan)
            return plan

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
            plan = await self._run_agent(run_id=run_id, input_text=input_text)

            self.on_answer_chunk(run_id=run_id, chunk=plan, idx='0')
            yield plan

        return _generator()

    def __str__(self) -> str:
        from io import StringIO
        import textwrap

        buf = StringIO()

        extra = []
        extra.append('Type: MAP Agent')

        extra_block = '\n'.join(extra)
        extra_block = textwrap.indent(extra_block, '\t')

        buf.write(super().__str__())
        buf.write(f'{extra_block}\n')

        return buf.getvalue()

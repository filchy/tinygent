from __future__ import annotations

import asyncio
from collections.abc import Sequence
import logging
from typing import AsyncGenerator
from typing import Literal
import uuid

from pydantic import Field

from tinygent.agents.base_agent import TinyBaseAgent
from tinygent.agents.base_agent import TinyBaseAgentConfig
from tinygent.core.datamodels.llm import AbstractLLM
from tinygent.core.datamodels.memory import AbstractMemory
from tinygent.core.datamodels.messages import AllTinyMessages
from tinygent.core.datamodels.messages import TinyChatMessage
from tinygent.core.datamodels.messages import TinyHumanMessage
from tinygent.core.datamodels.messages import TinySystemMessage
from tinygent.core.datamodels.middleware import AbstractMiddleware
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.runtime.executors import run_async_in_executor
from tinygent.core.telemetry.decorators import tiny_trace
from tinygent.core.telemetry.otel import set_tiny_attributes
from tinygent.core.types.base import TinyModel
from tinygent.core.types.io.llm_io_input import TinyLLMInput
from tinygent.prompts.map import MapPromptTemplate
from tinygent.prompts.map import get_prompt_template
from tinygent.utils.jinja_utils import render_template

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = get_prompt_template()


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

    @property
    def sum(self) -> str:
        return f'Next state({"valid" if self.is_valid else "non-valid"}): {self.next_state} reason: {self.reason} metadata: {self.metadata}'


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


class TinyMAPAgentConfig(TinyBaseAgentConfig['TinyMAPAgent']):
    """Configuration for the TinyMAPAgent."""

    type: Literal['map'] = Field(default='map')

    prompt_template: MapPromptTemplate = Field(_DEFAULT_PROMPT)

    max_plan_length: int = Field(default=4)

    max_branches_per_layer: int = Field(default=3)

    max_layer_depth: int = Field(default=2)

    max_recurrsion: int = Field(default=5)

    def build(self) -> TinyMAPAgent:
        return TinyMAPAgent(
            prompt_template=self.prompt_template,
            middleware=self.build_middleware_list(),
            llm=self.build_llm_instance(),
            tools=self.build_tools_list(),
            memory=self.build_memory_instance(),
            max_plan_length=self.max_plan_length,
            max_branches_per_layer=self.max_branches_per_layer,
            max_layer_depth=self.max_layer_depth,
            max_recurrsion=self.max_recurrsion,
        )


class TinyMAPAgent(TinyBaseAgent):
    """MAP (Monitoring, Action Proposal, Prediction) Agent implementation.

    Implements a sophisticated planning approach that decomposes complex tasks into
    subgoals and explores multiple action proposals for each subgoal through a
    search process. The agent uses monitoring to validate proposals, prediction to
    simulate outcomes, and evaluation to select the best action path.

    The MAP architecture consists of:
    - Task Decomposer: Breaks down complex tasks into manageable subgoals
    - Actor: Proposes actions to achieve subgoals
    - Monitor: Validates action proposals with feedback loops
    - Predictor: Simulates next states after actions
    - Evaluator: Scores states based on subgoal satisfaction
    - Orchestrator: Determines if subgoals are fully satisfied

    This agent performs tree search with configurable depth and branch factors,
    making it suitable for tasks requiring thorough exploration of action spaces.

    Middleware Hooks Activated:
    - before_llm_call / after_llm_call - For LLM calls
    - before_tool_call / after_tool_call - For tool executions
    - on_plan - When creating search/action plans
    - on_answer / on_answer_chunk - For final answers
    - on_error - On any error

    Note: MAP agent uses on_plan for action summaries but not on_reasoning or on_tool_reasoning.

    Args:
        llm: Language model for all MAP components
        memory: Memory system for maintaining conversation history
        max_plan_length: Maximum number of actions in final plan
        max_branches_per_layer: Number of action proposals to explore per layer
        max_layer_depth: Maximum depth of search tree exploration
        prompt_template: Template for MAP prompts (default provided)
        max_recurrsion: Maximum attempts to generate valid action proposals (default: 5)
        tools: List of tools available to the agent
        middleware: List of middleware to apply during execution
    """

    def __init__(
        self,
        llm: AbstractLLM,
        memory: AbstractMemory,
        max_plan_length: int,
        max_branches_per_layer: int,
        max_layer_depth: int,
        prompt_template: MapPromptTemplate = _DEFAULT_PROMPT,
        max_recurrsion: int = 5,
        tools: list[AbstractTool] = [],
        middleware: Sequence[AbstractMiddleware] = [],
    ) -> None:
        super().__init__(llm=llm, tools=tools, memory=memory, middleware=middleware)

        self.max_plan_length = max_plan_length
        self.max_branches_per_layer = max_branches_per_layer
        self.max_recurrsion = max_recurrsion
        self.max_layer_depth = max_layer_depth
        self.prompt_template = (
            prompt_template if prompt_template else self.default_prompt_template()
        )

    @tiny_trace('map_agent_task_decomposer')
    async def _task_decomposer(self, run_id: str, input_txt: str) -> list[str]:
        logger.debug('[TASK DECOMPOSER] task: %s', input_txt)

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

        result = await self.run_llm(
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
        logger.debug(
            '[TASK DECOMPOSER] decomposed task (%s): %s', input_txt, all_subgoals
        )
        return all_subgoals

    @tiny_trace('map_agent_actor')
    async def _actor(
        self,
        run_id: str,
        subgoal: str,
        prev_proposals: list[TinyMAPActionProposal],
        feedback: list[str],
    ) -> str:
        logger.debug(
            '[ACTOR] subgoal: %s prev proposals: %s feedback: %s',
            subgoal,
            [a.sum for a in prev_proposals],
            feedback,
        )

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

        result = await self.run_llm(
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
        logger.debug('[ACTOR] subgoal: %s subanswer: %s', subgoal, subanswer)
        return subanswer

    @tiny_trace('map_agent_monitor')
    async def _monitor(
        self,
        run_id: str,
        current_proposal: TinyMAPActionProposal,
        prev_proposals: list[TinyMAPActionProposal] = [],
    ) -> TinyMAPMonitorValidity:
        logger.debug(
            '[MONITOR] curr proposal: %s prev_proposals: %s',
            current_proposal.sum,
            [a.sum for a in prev_proposals],
        )

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

        result = await self.run_llm(
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
        logger.debug(
            '[MONITOR] curr proposal: %s valid: %r feedback: %s',
            current_proposal.sum,
            result.is_valid,
            result.feedback,
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
        logger.debug(
            '[SINGLE ACTION PROPOSAL] subgoal: %s prev_actions: %s',
            subgoal,
            [a.sum for a in prev_actions],
        )

        num_tries = 0
        validity = False

        feedback: list[str] = []
        all_proposals: list[TinyMAPActionProposal] = []

        while not validity and num_tries < self.max_recurrsion:
            subanswer = await self._actor(run_id, subgoal, prev_actions, feedback)

            proposal = TinyMAPActionProposal(question=subgoal, answer=subanswer)

            monitor_validity = await self._monitor(run_id, proposal, prev_actions)
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
        logger.debug(
            '[SINGLE ACTION PROPOSAL] subgoal: %s proposals: %s',
            subgoal,
            [p.sum for p in all_proposals],
        )

        return proposal

    @tiny_trace('map_agent.map.search.action_proposal')
    async def _action_proposal(
        self, run_id: str, subgoal: str, prev_actions: list[TinyMAPActionProposal]
    ) -> list[TinyMAPActionProposal]:
        logger.debug(
            '[ACTION PROPOSAL] subgoal: %s prev_actions: %s',
            subgoal,
            [a.sum for a in prev_actions],
        )

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
        logger.debug(
            '[ACTION PROPOSAL] subgoal: %s all proposals: %s',
            subgoal,
            formatted_proposals,
        )
        return proposals

    @tiny_trace('map_agent_predictor')
    async def _predictor(
        self, run_id: str, state: TinyMAPState, action: TinyMAPActionProposal
    ) -> TinyMAPState:
        logger.debug('[PREDICTOR] state: %s action: %s', state.sum, action.sum)

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

        result = await self.run_llm(
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
        logger.debug(
            '[PREDICTOR] state: %s action: %s result: %s',
            state.sum,
            action.sum,
            result.sum,
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
        logger.debug(
            '[SEARCH] depth: %d state: %s subgoal: %s', depth, state.sum, subgoal
        )

        eval_values: list[TinyMAPEvaluatorResult] = []
        next_states: list[TinyMAPState] = []
        actions: list[TinyMAPActionProposal] = []

        proposed_actions = await self._action_proposal(run_id, subgoal, [])

        for action in proposed_actions:
            logger.debug('[SEARCH] current action: %s', action.sum)
            pred_state = await self._predictor(run_id, state, action)

            orch_res = await self._orchestrator(run_id, pred_state, subgoal)

            if depth < self.max_layer_depth and not orch_res.fully_satisfies:
                child_res = await self._search(run_id, depth + 1, pred_state, subgoal)

                l_next_state = child_res.next_state
                l_eval_score = child_res.eval_score
                l_action = child_res.action
            else:
                l_next_state = pred_state
                l_eval_score = await self._evaluator(run_id, pred_state, subgoal)
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
        logger.debug(
            '[SEARCH] subgoal: %s best state: %s best action: %s best score: %d',
            subgoal,
            best_state.sum,
            best_action.sum,
            best_eval_score.score,
        )

        return TinyMAPSearchResult(
            next_state=best_state,
            action=best_action,
            eval_score=best_eval_score,
        )

    @tiny_trace('map_agent_evaluator')
    async def _evaluator(
        self, run_id: str, state: TinyMAPState, subgoal: str
    ) -> TinyMAPEvaluatorResult:
        logger.debug('[EVALUATOR] subgoal: %s state: %s', subgoal, state.sum)

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

        result = await self.run_llm(
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
        logger.debug(
            '[EVALUATOR] subgoal: %s state: %s score: %d',
            subgoal,
            state.sum,
            result.score,
        )
        return result

    @tiny_trace('map_agent_orchestrator')
    async def _orchestrator(
        self, run_id: str, state: TinyMAPState, subgoal: str
    ) -> TinyMAPOrchestratorResult:
        logger.debug('[ORCHESTRATOR] subgoal: %s current state: %s', subgoal, state.sum)

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

        result = await self.run_llm(
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
        logger.debug('[ORCHESTRATOR] finished with %r', result.fully_satisfies)
        return result

    @tiny_trace('map_agent_map')
    async def _map(self, run_id: str, question: str) -> list[TinyMAPActionProposal]:
        logger.debug('[MAP] Running MAP module with task: %s', question)
        subgoals = await self._task_decomposer(run_id, question)
        subgoals.append(
            question
        )  # INFO: Last and final subgoal is original user question

        final_plan: list[TinyMAPActionProposal] = []

        for subgoal in subgoals:
            logger.debug('[MAP] current subgoal: %s', subgoal)
            current_state = TinyMAPState(
                is_valid=True,
                next_state=f'Initial problem context: {question}',
                reason='initial state',
                metadata='',
            )

            validity = await self._orchestrator(run_id, current_state, subgoal)

            while (
                not validity.fully_satisfies and len(final_plan) < self.max_plan_length
            ):
                search_res = await self._search(run_id, 0, current_state, subgoal)

                final_plan.append(search_res.action)
                current_state = search_res.next_state
                validity = await self._orchestrator(run_id, current_state, subgoal)

                await self.on_plan(run_id=run_id, plan=search_res.action.sum, kwargs={})

        set_tiny_attributes(
            {'agent.map.final_plan': '\n'.join([p.sum for p in final_plan])}
        )
        logger.debug(
            '[MAP] task: %s final plan: %s', question, [p.sum for p in final_plan]
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
        logger.debug('Running agent with task: %s', input_text)

        self.memory.save_context(TinyHumanMessage(content=input_text))

        try:
            final_plan = await self._map(run_id, input_text)

            return '\n\n'.join([p.sum for p in final_plan])
        except Exception as e:
            logger.warning('Error during MAP: %s', e)
            await self.on_error(run_id=run_id, e=e, kwargs={})
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

            await self.on_answer(run_id=run_id, answer=plan, kwargs={})
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

            await self.on_answer_chunk(run_id=run_id, chunk=plan, idx='0', kwargs={})
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

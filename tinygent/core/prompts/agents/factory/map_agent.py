from tinygent.core.prompt import TinyPrompt
from tinygent.core.prompts.agents.template.map_agent import ActionProposalPromptTemplate
from tinygent.core.prompts.agents.template.map_agent import ActorPromptTemplate
from tinygent.core.prompts.agents.template.map_agent import MapPromptTemplate
from tinygent.core.prompts.agents.template.map_agent import MonitorPrompTemplate
from tinygent.core.prompts.agents.template.map_agent import OrchestratorPromptTemplate
from tinygent.core.prompts.agents.template.map_agent import PredictorPromptTemplate
from tinygent.core.prompts.agents.template.map_agent import TaskDecomposerPromptTemplate


def get_prompt_template() -> MapPromptTemplate:
    return MapPromptTemplate(
        task_decomposer=TaskDecomposerPromptTemplate(
            system='You are an AI assistant specializing in decomposing complex tasks into simpler sub-tasks. Your role is to break down questions into a set of manageable sub-questions that can be answered sequentially.',
            user='Question: {{ question }}\n\nBreak down the above question into a set of simpler sub-questions that need to be answered in order to solve the main question. Pay close attention to the exact criteria required. Think step by step about the knowledge required, then provide a list of sub-questions. Maximum number of sub-questions is {{ max_subquestions }}.',
        ),
        action_proposal=ActionProposalPromptTemplate(
            actor=ActorPromptTemplate(
                init=TinyPrompt.UserSystem(
                    system='You are an AI assistant that provides detailed and accurate answers to questions.',
                    user='Question: {{ question }}\n\nPlease provide an answer to the above question. Think carefully before answering. You must provide an answer even if you are uncertain. The question may pertain to hypothetical or counterfactual scenarios.',
                ),
                init_fixer=TinyPrompt.UserSystem(
                    system='You are an AI assistant that refines and improves answers based on validation feedback.',
                    user='{{ validation }}\n\nQuestion: {{ question }}\n\nPlease try again to provide an answer to the above question based on the validation feedback. Think carefully, then provide your improved answer. You must provide an answer even if you are uncertain.',
                ),
                continuos=TinyPrompt.UserSystem(
                    system='You are an AI assistant that answers questions while considering previously answered sub-questions.',
                    user='Our goal is to answer the following question:\n\nQuestion: {{ question }}\n\nTo answer this question, we need to consider the following sub-questions and their answers:\n\n{{ previous_questions }}\n\nBased on the above information, please provide an answer to the original question. Think carefully about the question and the provided sub-questions and answers before responding. You must provide an answer even if you are uncertain.',
                ),
                continuos_fixer=TinyPrompt.UserSystem(
                    system='You are an AI assistant that refines answers based on validation feedback while considering context from previous sub-questions.',
                    user='{{ validation }}\n\nQuestion: {{ question }}\n\nPlease try again to provide an answer to the above question based on the validation feedback. Think carefully, then provide your improved answer. You must provide an answer even if you are uncertain.',
                ),
                evaluator=TinyPrompt.UserSystem(
                    system='You are an evaluator that scores how well a predicted state achieves a given subgoal. Always output a single integer between 0 and 100, where 100 means the subgoal is fully achieved and 0 means no progress. Higher scores represent better progress. If the state is invalid, return 0.',
                    user='Evaluate how close the following predicted state is to achieving the subgoal.\n\nState:\n{{ state }}\n\nSubgoal:\n{{ subgoal }}\n\nReturn a single integer score from 0 to 100.',
                ),
            ),
            monitor=MonitorPrompTemplate(
                init=TinyPrompt.UserSystem(
                    system='You are an AI assistant that validates whether proposed answers correctly address the given questions.',
                    user='Our goal is to answer the following question:\n\nQuestion: {{ question }}\n\nThe following answer has been proposed:\n\nProposed Answer: {{ answer }}\n\nIs this the correct answer to the original question? Think carefully about the question and the proposed answer. Indicate whether the proposed answer is correct. If the proposed answer does not contain a final answer, indicate it as invalid.',
                ),
                continuos=TinyPrompt.UserSystem(
                    system='You are an AI assistant that validates whether proposed answers correctly address questions, considering context from previously answered sub-questions.',
                    user='Our goal is to answer the following question:\n\nQuestion: {{ question }}\n\nTo answer this question, we first considered these sub-questions and their answers:\n\n{{ previous_questions }}\n\nBased on this information, the following answer has been proposed for the original question:\n\nProposed Answer: {{ answer }}\n\nIs this the correct answer to the original question? Think carefully about the question, the proposed answer, and the provided sub-questions and answers. Indicate whether the proposed answer is correct. If the proposed answer does not contain a final answer, indicate it as invalid.',
                ),
            ),
        ),
        predictor=PredictorPromptTemplate(
            user='Current state:\n{{ state }}\n\nProposed action:\n{{ proposed_action }}\n\nTask:\n- Determine whether the action can be applied\n- If it can, produce the resulting next state\n- If it cannot, set "is_valid" to false and explain the reason\n\nProvide your prediction in the required format with fields: is_valid, next_state, reason, and metadata.',
            system='You are a state transition model that simulates how the environment changes when an action is applied. Your output must always describe the next state after applying the action, including whether the transition is valid, what the next state would be, the reasoning, and any relevant metadata.',
        ),
        orchestrator=OrchestratorPromptTemplate(
            user='We are solving the following question:\n\nQuestion: {{ question }}\n\nThe following answer has been proposed:\n\nProposed Answer: {{ answer }}\n\nYour task: Determine whether this proposed answer fully satisfies the question. Before responding, carefully examine the wording and criteria of the question, whether the proposed answer directly addresses it, whether it is complete, correct, and unambiguous, and whether it is logically consistent.\n\nDoes this proposed answer fully satisfy the question?',
            system='You are an AI assistant expert at determining if an answer actually answers the question. You assess whether proposed answers fully satisfy the given questions by checking completeness, correctness, and logical consistency.',
        ),
    )

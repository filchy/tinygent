import asyncio

import seznam.tools  # noqa: F401

from tinygent.agents.multi_step_agent import ActionPromptTemplate
from tinygent.agents.multi_step_agent import FinalAnswerPromptTemplate
from tinygent.agents.multi_step_agent import PlanPromptTemplate
from tinygent.agents.multi_step_agent import MultiStepPromptTemplate
from tinygent.agents.multi_step_agent import TinyMultiStepAgent
from tinygent.llms.openai import OpenAIConfig
from tinygent.llms.openai import OpenAILLM
from tinygent.logging import setup_general_loggers
from tinygent.logging import setup_logger
from tinygent.memory.buffer_chat_memory import BufferChatMemory
from tinygent.runtime.global_registry import GlobalRegistry
from tinygent.utils.load_file import load_yaml

logger = setup_logger('debug')
setup_general_loggers('warning')


async def main():
    tools = GlobalRegistry.get_registry().get_tools()
    agent_prompt = load_yaml('seznam/prompts/agent-prompt.yaml')

    multi_step_agent = TinyMultiStepAgent(
        llm=OpenAILLM(
            config=OpenAIConfig(
                model_name='azure-gpt-4.1-nano',
                timeout=10.0,
            )
        ),
        memory_list=[BufferChatMemory()],
        tools=list(tools.values()),
        prompt_template=MultiStepPromptTemplate(
            acter=ActionPromptTemplate(
                system=agent_prompt['acter']['system'],
                final_answer=agent_prompt['acter']['final_answer'],
            ),
            plan=PlanPromptTemplate(
                init_plan=agent_prompt['planner']['init_plan'],
                update_plan=agent_prompt['planner']['update_plan'],
            ),
            final=FinalAnswerPromptTemplate(
                final_answer=agent_prompt['final']['final_answer']
            ),
        ),
    )

    result = multi_step_agent.run(
        # 'Chci si na svém mobilu zahrát Hansu teutonicu. Mám android'
        # 'Jaký je zdravotní stav papeže Františka?',
        'Jsou nějaké negativní zkušenosti s produktem Marshall Major IV BT, černá?'
    )

    logger.info(f'[RESULT] {result}')


if __name__ == '__main__':
    asyncio.run(main())

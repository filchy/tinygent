from pathlib import Path

from tinygent.agents.map_agent import MapPromptTemplate
from tinygent.agents.map_agent import TinyMAPAgent
from tinygent.llms.base import init_llm
from tinygent.logging import setup_logger
from tinygent.memory.buffer_chat_memory import BufferChatMemory
from tinygent.utils.yaml import tiny_yaml_load

logger = setup_logger('debug')


async def main():
    map_agent_prompt = tiny_yaml_load(str(Path(__file__).parent / 'prompts.yaml'))

    agent = TinyMAPAgent(
        llm=init_llm('openai:gpt-4o-mini', temperature=0.1),
        prompt_template=MapPromptTemplate(**map_agent_prompt),
        memory=BufferChatMemory(),
        max_plan_length=4,
        max_branches_per_layer=3,
        max_layer_depth=4,
        max_recurrsion=3,
    )

    result = agent.run(
        'Will the Albany in Georgia reach a hundred thousand occupants before the one in New York?'
    )

    logger.info(f'[RESULT] {result}')
    logger.info(f'[AGENT] {str(agent)}')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())

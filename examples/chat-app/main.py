import uuid
from pathlib import Path

import tiny_chat as tc
from tinygent.cli.builder import build_agent
from tinygent.cli.utils import discover_and_register_components
from tinygent.logging import setup_general_loggers
from tinygent.logging import setup_logger
from tinygent.utils.yaml import tiny_yaml_load

logger = setup_logger('debug')
setup_general_loggers('warning')
discover_and_register_components()


async def answer_hook(ans: str):
    await tc.BaseMessage(
        id=str(uuid.uuid4()),
        type='text',
        sender='agent',
        content=ans,
    ).send()


async def answer_chunk_hook(ans_chunk: str, id: str):
    await tc.BaseMessage(
        id=id,
        type='text',
        sender='agent',
        content=ans_chunk,
    ).send()

agent = build_agent(
    tiny_yaml_load(str(Path(__file__).parent.parent / 'agents' / 'react' / 'agent.yaml'))
)
agent.on_answer = answer_hook
agent.on_answer_chunk = answer_chunk_hook


@tc.on_message
async def handle_message(msg: tc.BaseMessage):
    async for _ in agent.run_stream(msg.content):
        pass


if __name__ == '__main__':
    tc.run(reload=True, log_level='debug', access_log=False)

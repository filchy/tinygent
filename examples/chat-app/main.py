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


# TODO: finish all the hooks (error, tool call, streaming chunks, etc.)
async def answer_hook(ans: str):
    await tc.BaseMessage(
        id='answer-hook-msg',
        type='text',
        sender='agent',
        content=ans,
        streaming=False,
    ).send()


agent = build_agent(
    tiny_yaml_load(str(Path(__file__).parent.parent / 'agents' / 'react' / 'agent.yaml'))
)
agent.on_answer = answer_hook


@tc.on_message
def handle_message(msg: tc.BaseMessage):
    agent.run(msg.content)


if __name__ == '__main__':
    tc.run(reload=True, log_level='debug', access_log=False)

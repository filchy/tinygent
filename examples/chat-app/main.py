import uuid
from pathlib import Path
from typing import Any

import tiny_chat as tc
from tinygent.cli.builder import build_agent
from tinygent.cli.utils import discover_and_register_components
from tinygent.datamodels.tool import AbstractTool
from tinygent.logging import setup_general_loggers
from tinygent.logging import setup_logger
from tinygent.utils.yaml import tiny_yaml_load

logger = setup_logger('debug')
setup_general_loggers('warning')
discover_and_register_components()


async def answer_hook(*, run_id: str, answer: str):
    await tc.BaseMessage(
        id=run_id,
        type='text',
        sender='agent',
        content=answer,
    ).send()


async def answer_chunk_hook(*, run_id: str, chunk: str, idx: str):
    await tc.BaseMessage(
        id=run_id,
        type='text',
        sender='agent',
        content=chunk,
    ).send()


async def tool_call_hook(*, run_id: str, tool: AbstractTool, args: dict[str, Any], result: Any):
    await tc.AgentToolCallMessage(
        id=str(uuid.uuid4()),
        parent_id=run_id,
        tool_name=tool.info.name,
        tool_args=args,
    ).send()


agent = build_agent(
    tiny_yaml_load(str(Path(__file__).parent.parent / 'agents' / 'react' / 'agent.yaml'))
)
agent.on_answer = answer_hook
agent.on_answer_chunk = answer_chunk_hook
agent.on_after_tool_call = tool_call_hook


@tc.on_message
async def handle_message(msg: tc.BaseMessage):
    async for _ in agent.run_stream(msg.content):
        pass


if __name__ == '__main__':
    tc.run(reload=True, log_level='debug', access_log=False)

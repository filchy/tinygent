import asyncio
from io import StringIO
import textwrap
from typing import Literal

from pydantic import Field

from tinygent.core.datamodels.memory import AbstractMemory
from tinygent.core.datamodels.memory import AbstractMemoryConfig
from tinygent.core.datamodels.messages import AllTinyMessages
from tinygent.core.factory.memory import build_memory
from tinygent.memory.base_chat_memory import BaseChatMemory


class CombinedMemoryConfig(AbstractMemoryConfig['CombinedMemory']):
    type: Literal['combined'] = Field(default='combined', frozen=True)

    memory_list: list[AbstractMemoryConfig] = []

    def build(self) -> 'CombinedMemory':
        memories = [build_memory(cfg) for cfg in self.memory_list]
        return CombinedMemory(memory_list=memories)


class CombinedMemory(BaseChatMemory):
    """Composite memory combining multiple memory systems.

    Coordinates multiple memory instances, allowing different memory strategies
    to work together. Messages are saved to all constituent memories, and
    variables are loaded and merged from all memories.

    This enables sophisticated memory architectures such as:
    - Separate memories for different information types
    - Combining window and summary memories
    - Multi-tier memory with different retention policies

    Operations (save, load, clear) are propagated to all constituent memories,
    with async operations executed in parallel for efficiency.

    Args:
        memory_list: List of memory instances to combine
    """

    def __init__(self, memory_list: list[AbstractMemory]) -> None:
        super().__init__()

        self.memory_list: list[AbstractMemory] = memory_list

    @property
    def memory_keys(self) -> list[str]:
        keys = []
        for memory in self.memory_list:
            keys.extend(memory.memory_keys)
        return keys

    def load_variables(self) -> dict[str, str]:
        memory_vars = {}
        for memory in self.memory_list:
            memory_vars.update(memory.load_variables())
        return memory_vars

    def save_context(self, message: AllTinyMessages) -> None:
        self._chat_history.add_message(message)

        for memory in self.memory_list:
            memory.save_context(message)

    def clear(self) -> None:
        for memory in self.memory_list:
            memory.clear()

    async def aload_variables(self) -> dict[str, str]:
        results = await asyncio.gather(
            *[memory.aload_variables() for memory in self.memory_list]
        )
        memory_vars = {}
        for r in results:
            memory_vars.update(r)
        return memory_vars

    async def asave_context(self, message: AllTinyMessages) -> None:
        await asyncio.gather(
            *[memory.asave_context(message) for memory in self.memory_list]
        )

    async def aclear(self) -> None:
        await asyncio.gather(*[memory.aclear() for memory in self.memory_list])

    def __str__(self) -> str:
        buff = StringIO()

        buff.write('type: Memories\n')
        buff.write(f'Combined Memories ({len(self.memory_list)}):\n')
        for memory in self.memory_list:
            buff.write(f'{textwrap.indent(str(memory), "\t")}\n')

        return buff.getvalue()

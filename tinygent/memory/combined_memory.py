import asyncio
from io import StringIO
import textwrap
from typing import Literal

from tinygent.cli.builder import build_memory
from tinygent.datamodels.memory import AbstractMemory
from tinygent.datamodels.memory import AbstractMemoryConfig
from tinygent.datamodels.messages import AllTinyMessages
from tinygent.memory.base_chat_memory import BaseChatMemory


class CombinedMemoryConfig(AbstractMemoryConfig['CombinedMemory']):
    type: Literal['combined'] = 'combined'

    memory_list: list[AbstractMemoryConfig]

    def build(self) -> 'CombinedMemory':
        memories = [build_memory(cfg) for cfg in self.memory_list]
        return CombinedMemory(memory_list=memories)


class CombinedMemory(BaseChatMemory):
    memory_list: list[AbstractMemory]

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

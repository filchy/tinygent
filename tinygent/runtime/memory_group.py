import asyncio

from tinygent.datamodels.memory import AbstractMemory
from tinygent.datamodels.messages import AllTinyMessages


class MemoryGroup(AbstractMemory):
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

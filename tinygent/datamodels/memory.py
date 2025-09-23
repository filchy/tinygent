from abc import ABC
from abc import abstractmethod

from pydantic import BaseModel

from tinygent.datamodels.messages import AllTinyMessages
from tinygent.runtime.executors import run_sync_in_executor


class AbstractMemory(BaseModel, ABC):
    @property
    @abstractmethod
    def memory_keys(self) -> list[str]: ...

    @abstractmethod
    def load_variables(self) -> dict[str, str]: ...

    @abstractmethod
    def save_context(self, message: AllTinyMessages) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...

    async def aload_variables(self) -> dict[str, str]:
        return await run_sync_in_executor(self.load_variables)

    async def asave_context(self, message: AllTinyMessages) -> None:
        return await run_sync_in_executor(self.save_context, message)

    async def aclear(self) -> None:
        return await run_sync_in_executor(self.clear)

from abc import ABC
from abc import abstractmethod

from pydantic import BaseModel

from tinygent.datamodels.llm_io import TinyLLMInput
from tinygent.datamodels.llm_io import TinyLLMResult
from tinygent.runtime.executors import run_in_executor


class AbstractMemory(BaseModel, ABC):

    @property
    @abstractmethod
    def memory_keys(self) -> list[str]: ...

    @abstractmethod
    def load_variables(self) -> dict[str, str]: ...

    @abstractmethod
    def save_context(
        self,
        input: TinyLLMInput,
        output: TinyLLMResult
    ) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...

    async def aload_variables(self) -> dict[str, str]:

        return await run_in_executor(self.load_variables)

    async def asave_context(
        self,
        input: TinyLLMInput,
        output: TinyLLMResult
    ) -> None:

        return await run_in_executor(self.save_context, input, output)

    async def aclear(self) -> None:

        return await run_in_executor(self.clear)

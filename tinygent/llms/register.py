from tinygent.llms.openai import OpenAIConfig
from tinygent.llms.openai import OpenAILLM
from tinygent.runtime.global_registry import GlobalRegistry


def _register_llms() -> None:
    registry = GlobalRegistry().get_registry()

    registry.register_llm('openai', OpenAIConfig, OpenAILLM)


_register_llms()

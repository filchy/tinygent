from tiny_mistralai.mistralai import MistralAIConfig
from tiny_mistralai.mistralai import MistralAILLM
from tinygent.runtime.global_registry import GlobalRegistry


def _register_mistralai() -> None:
    registry = GlobalRegistry().get_registry()

    registry.register_llm('mistralai', MistralAIConfig, MistralAILLM)


_register_mistralai()

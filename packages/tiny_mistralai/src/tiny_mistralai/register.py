def _register_mistralai() -> None:
    from tinygent.runtime.global_registry import GlobalRegistry

    from .llm import MistralAILLM
    from .llm import MistralAILLMConfig

    registry = GlobalRegistry().get_registry()

    registry.register_llm('mistralai', MistralAILLMConfig, MistralAILLM)


_register_mistralai()

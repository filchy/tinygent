def _register_mistralai() -> None:
    from tinygent.core.runtime.global_registry import GlobalRegistry

    from .embedder import MistralAIEmbedder
    from .embedder import MistralAIEmbedderConfig
    from .llm import MistralAILLM
    from .llm import MistralAILLMConfig

    registry = GlobalRegistry().get_registry()

    registry.register_llm('mistralai', MistralAILLMConfig, MistralAILLM)
    registry.register_embedder('mistralai', MistralAIEmbedderConfig, MistralAIEmbedder)


_register_mistralai()

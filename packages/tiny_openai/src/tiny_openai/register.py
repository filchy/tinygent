def _register_openai() -> None:
    from tinygent.core.runtime.global_registry import GlobalRegistry

    from .embedder import OpenAIEmbedder
    from .embedder import OpenAIEmbedderConfig
    from .llm import OpenAILLM
    from .llm import OpenAILLMConfig

    registry = GlobalRegistry().get_registry()

    registry.register_llm('openai', OpenAILLMConfig, OpenAILLM)
    registry.register_embedder('openai', OpenAIEmbedderConfig, OpenAIEmbedder)


_register_openai()

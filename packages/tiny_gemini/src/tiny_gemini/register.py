def _register_gemini() -> None:
    from tinygent.core.runtime.global_registry import GlobalRegistry

    from .llm import GeminiLLM
    from .llm import GeminiLLMConfig

    registry = GlobalRegistry().get_registry()

    registry.register_llm('gemini', GeminiLLMConfig, GeminiLLM)


_register_gemini()

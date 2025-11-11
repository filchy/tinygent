from tiny_gemini.gemini import GeminiConfig
from tiny_gemini.gemini import GeminiLLM
from tinygent.runtime.global_registry import GlobalRegistry


def _register_gemini() -> None:
    registry = GlobalRegistry().get_registry()

    registry.register_llm('gemini', GeminiConfig, GeminiLLM)


_register_gemini()

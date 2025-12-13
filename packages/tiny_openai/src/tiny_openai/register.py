from tiny_openai.embedder import OpenAIEmbedder
from tiny_openai.embedder import OpenAIEmbedderConfig
from tiny_openai.llm import OpenAILLM
from tiny_openai.llm import OpenAILLMConfig
from tinygent.runtime.global_registry import GlobalRegistry


def _register_openai() -> None:
    registry = GlobalRegistry().get_registry()

    registry.register_llm('openai', OpenAILLMConfig, OpenAILLM)
    registry.register_embedder('openai', OpenAIEmbedderConfig, OpenAIEmbedder)


_register_openai()

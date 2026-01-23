def _register_anthropic() -> None:
    from tinygent.core.runtime.global_registry import GlobalRegistry

    from .llm import ClaudeLLM
    from .llm import ClaudeLLMConfig

    registry = GlobalRegistry().get_registry()

    registry.register_llm('anthropic', ClaudeLLMConfig, ClaudeLLM)


_register_anthropic()

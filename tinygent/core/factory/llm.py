from tinygent.core.datamodels.llm import AbstractLLM
from tinygent.core.datamodels.llm import AbstractLLMConfig
from tinygent.core.factory.helper import check_modules
from tinygent.core.factory.helper import parse_config
from tinygent.core.factory.helper import parse_model
from tinygent.core.runtime.global_registry import GlobalRegistry


def build_llm(
    llm: str | dict | AbstractLLMConfig,
    *,
    provider: str | None = None,
    temperature: float | None = None,
    **kwargs,
) -> AbstractLLM:
    """Build tiny llm."""
    check_modules()

    if isinstance(llm, str):
        model_provider, model_name = parse_model(llm, provider)

        llm_dict = {'model': model_name, 'type': model_provider, **kwargs}

        if temperature:
            llm_dict['temperature'] = temperature

        if model_provider == 'openai':
            from tiny_openai import OpenAILLMConfig

            return OpenAILLMConfig(**llm_dict).build()
        elif model_provider == 'mistralai':
            from tiny_mistralai import MistralAILLMConfig

            return MistralAILLMConfig(**llm_dict).build()
        elif model_provider == 'gemini':
            from tiny_gemini import GeminiLLMConfig

            return GeminiLLMConfig(**llm_dict).build()
        elif model_provider == 'anthropic':
            from tiny_anthropic import ClaudeLLMConfig

            return ClaudeLLMConfig(**llm_dict).build()
        else:
            raise ValueError(f'Unsupported model provider: {model_provider}')

    if isinstance(llm, AbstractLLMConfig):
        llm = llm.model_dump()

    llm_config = parse_config(llm, lambda: GlobalRegistry.get_registry().get_llms())
    return llm_config.build()

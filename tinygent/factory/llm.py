from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.llm import AbstractLLMConfig
from tinygent.factory.helper import check_modules
from tinygent.factory.helper import parse_config
from tinygent.runtime.global_registry import GlobalRegistry


def parse_model(model: str, model_provider: str | None = None) -> tuple[str, str]:
    if ':' not in model and model_provider is None:
        raise ValueError(
            'Model string must be in the format "model_provider:model_name" '
            'or a model_provider must be specified.'
        )

    if model_provider is None:
        model_provider, model = model.split(':')

    return model_provider, model


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

        llm_dict = {'model': model_name, **kwargs}

        if temperature:
            llm_dict['temperature'] = temperature

        if model_provider == 'openai':
            from tiny_openai import OpenAIConfig

            return OpenAIConfig(**llm_dict).build()
        elif model_provider == 'mistralai':
            from tiny_mistralai import MistralAIConfig

            return MistralAIConfig(**llm_dict).build()
        elif model_provider == 'gemini':
            from tiny_gemini import GeminiConfig

            return GeminiConfig(**llm_dict).build()
        else:
            raise ValueError(f'Unsupported model provider: {model_provider}')

    if isinstance(llm, AbstractLLMConfig):
        llm = llm.model_dump()

    llm_config = parse_config(llm, lambda: GlobalRegistry.get_registry().get_llms())
    return llm_config.build()

from tinygent.datamodels.llm import AbstractLLM


def _parse_model(model: str, model_provider: str | None = None) -> tuple[str, str]:
    if ':' not in model and model_provider is None:
        raise ValueError(
            'Model string must be in the format "model_provider:model_name" '
            'or a model_provider must be specified.'
        )

    if model_provider is None:
        model_provider, model = model.split(':')

    return model_provider, model


def init_llm(model: str, *, model_provider: str | None = None, **kwargs) -> AbstractLLM:
    model_provider, model = _parse_model(model, model_provider)

    if model_provider == 'openai':
        from tiny_openai import OpenAIConfig

        return OpenAIConfig(model=model, **kwargs).build()
    else:
        raise ValueError(f'Unsupported model provider: {model_provider}')

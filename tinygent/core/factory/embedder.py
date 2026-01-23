from tinygent.core.datamodels.embedder import AbstractEmbedder
from tinygent.core.datamodels.embedder import AbstractEmbedderConfig
from tinygent.core.factory.helper import check_modules
from tinygent.core.factory.helper import parse_config
from tinygent.core.factory.helper import parse_model
from tinygent.core.runtime.global_registry import GlobalRegistry


def build_embedder(
    embedder: str | dict | AbstractEmbedderConfig,
    *,
    provider: str | None = None,
    **kwargs,
) -> AbstractEmbedder:
    """Builc tiny embedder."""
    check_modules()

    if isinstance(embedder, str):
        model_provider, model_name = parse_model(embedder, provider)

        embed_dict = {'model': model_name, 'type': model_provider, **kwargs}

        if model_provider == 'openai':
            from tiny_openai import OpenAIEmbedderConfig

            return OpenAIEmbedderConfig(**embed_dict).build()
        elif model_provider == 'mistralai':
            from tiny_mistralai import MistralAIEmbedderConfig

            return MistralAIEmbedderConfig(**embed_dict).build()
        elif model_provider == 'gemini':
            from tiny_gemini import GeminiEmbedderConfig

            return GeminiEmbedderConfig(**embed_dict).build()
        elif model_provider == 'voyageai':
            from tiny_voyageai import VoyageAIEmbedderConfig

            return VoyageAIEmbedderConfig(**embed_dict).build()
        else:
            raise ValueError(f'Unsupported model provider: {model_provider}')

    if isinstance(embedder, AbstractEmbedderConfig):
        embedder = embedder.model_dump()

    embedder_cfg = parse_config(
        embedder, lambda: GlobalRegistry.get_registry().get_embedders()
    )
    return embedder_cfg.build()

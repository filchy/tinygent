import logging
from typing import overload

from tinygent.datamodels.cross_encoder import AbstractCrossEncoder
from tinygent.datamodels.cross_encoder import AbstractCrossEncoderConfig
from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.llm import AbstractLLMConfig
from tinygent.factory.helper import check_modules
from tinygent.factory.helper import parse_config
from tinygent.runtime.global_registry import GlobalRegistry

logger = logging.getLogger(__name__)


@overload
def build_cross_encoder(
    crossencoder: dict | AbstractCrossEncoderConfig,
) -> AbstractCrossEncoder: ...


@overload
def build_cross_encoder(
    crossencoder: dict | AbstractCrossEncoderConfig | str,
    *,
    score_range: tuple[float, float] | None = None,
    llm: dict | AbstractLLM | AbstractLLMConfig | str | None = None,
) -> AbstractCrossEncoder: ...


@overload
def build_cross_encoder(
    crossencoder: dict | AbstractCrossEncoderConfig | str,
    *,
    score_range: tuple[float, float] | None = None,
    llm: dict | AbstractLLM | AbstractLLMConfig | str | None = None,
    llm_provider: str | None = None,
    llm_temperature: float | None = None,
) -> AbstractCrossEncoder: ...


def build_cross_encoder(
    crossencoder: dict | AbstractCrossEncoderConfig | str,
    *,
    score_range: tuple[float, float] | None = None,
    llm: dict | AbstractLLM | AbstractLLMConfig | str | None = None,
    llm_provider: str | None = None,
    llm_temperature: float | None = None,
    **kwargs,
) -> AbstractCrossEncoder:
    """Build tiny cross-encoder."""
    check_modules()

    if isinstance(crossencoder, str):
        crossencoder = {'type': crossencoder, **kwargs}

    if isinstance(crossencoder, AbstractCrossEncoderConfig):
        crossencoder = crossencoder.model_dump()

    if llm:
        from tinygent.factory.llm import build_llm

        if crossencoder.get('llm'):
            logger.warning('Overwriting existing agents llm with new one.')

        crossencoder['llm'] = (
            llm
            if isinstance(llm, AbstractLLM)
            else build_llm(llm, provider=llm_provider, temperature=llm_temperature)
        )

    if score_range:
        if crossencoder.get('score_range'):
            logger.warning('Overwriting existing score_range llm with new one.')
            crossencoder['score_range'] = score_range

    crossencoder_config = parse_config(
        crossencoder, lambda: GlobalRegistry.get_registry().get_crossencoders()
    )

    return crossencoder_config.build()

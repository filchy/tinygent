import json
from typing import Any
from typing import Iterable

from tinygent.core.datamodels.cross_encoder import AbstractCrossEncoderConfig
from tinygent.core.datamodels.embedder import AbstractEmbedderConfig
from tinygent.core.datamodels.llm import AbstractLLMConfig
from tinygent.core.datamodels.messages import AllTinyMessages
from tinygent.core.datamodels.tool import AbstractTool
from tinygent.core.telemetry.otel import set_tiny_attributes


def set_embedder_telemetry_attributes(
    config: AbstractEmbedderConfig,
    query: str | list[str],
    *,
    embedding_dim: int,
    result_len: int | None = None,
) -> None:
    """Unified telemetry attribute setter for all embedder methods."""
    queries = [query] if isinstance(query, str) else query
    attrs: dict[str, Any] = {
        'model.config': json.dumps(config.model_dump(mode='json')),
        'embedding.dim': embedding_dim,
        'queries': queries,
        'queries.len': len(queries),
    }

    if result_len is not None:
        attrs['result.len'] = result_len

    set_tiny_attributes(attrs)  # type: ignore[arg-type]


def set_llm_telemetry_attributes(
    config: AbstractLLMConfig,
    messages: Iterable[AllTinyMessages],
    *,
    result: str | list[str] | None = None,
    tools: list[AbstractTool] | None = None,
    output_schema: type | None = None,
) -> None:
    """Unified telemetry attribute setter for all LLM methods."""
    attrs: dict[str, Any] = {
        'model.config': json.dumps(config.model_dump(mode='json')),
        'messages': [m.tiny_str for m in messages],
        'messages.len': len(list(messages)),
    }

    if tools is not None:
        attrs['tools'] = [tool.info.name for tool in tools]
        attrs['tools.len'] = len(tools)

    if output_schema is not None:
        attrs['output_schema'] = output_schema.__name__

    if result is not None:
        attrs['result'] = result

    set_tiny_attributes(attrs)  # type: ignore[arg-type]


def set_cross_encoder_telemetry_attributes(
    config: AbstractCrossEncoderConfig,
    *,
    query: str | None = None,
    texts: Iterable[str] | None = None,
    pairs: Iterable[tuple[str, str]] | None = None,
    result: list[tuple[tuple[str, str], float]] | None = None,
) -> None:
    """Unified telemetry attribute setter for all Cross-encoder methods."""
    attrs: dict[str, Any] = {
        'model.config': json.dumps(config.model_dump(mode='json')),
    }

    if query is not None:
        attrs['query'] = query

    if texts is not None:
        texts_list = list(texts)
        attrs['texts'] = texts_list
        attrs['texts.len'] = len(texts_list)

    if pairs is not None:
        pairs_list = list(pairs)
        attrs['pairs'] = [[p[0], p[1]] for p in pairs_list]
        attrs['pairs.len'] = len(pairs_list)

    if result is not None:
        attrs['result'] = json.dumps(
            [{'query': r[0][0], 'text': r[0][1], 'score': r[1]} for r in result]
        )
        attrs['result.scores'] = [r[1] for r in result]
        attrs['result.len'] = len(result)

    set_tiny_attributes(attrs)  # type: ignore[arg-type]

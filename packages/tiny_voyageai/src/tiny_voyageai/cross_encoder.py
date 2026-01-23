from __future__ import annotations

from collections import defaultdict
import os
from typing import Iterable
from typing import Literal

from pydantic import Field
from pydantic import SecretStr
from voyageai.client_async import AsyncClient

from tinygent.core.datamodels.cross_encoder import AbstractCrossEncoder
from tinygent.core.datamodels.cross_encoder import AbstractCrossEncoderConfig
from tinygent.core.runtime.executors import run_in_semaphore
from tinygent.core.telemetry.decorators import tiny_trace
from tinygent.core.telemetry.utils import set_cross_encoder_telemetry_attributes


class VoyageAICrossEncoderConfig(AbstractCrossEncoderConfig['VoyageAICrossEncoder']):
    type: Literal['voyageai'] = Field(default='voyageai', frozen=True)

    model: str = Field(default='rerank-2.5')

    api_key: SecretStr | None = Field(
        default_factory=lambda: (
            SecretStr(os.environ['VOYAGE_API_KEY'])
            if 'VOYAGE_API_KEY' in os.environ
            else None
        ),
    )

    base_url: str | None = Field(default=None)

    timeout: float = Field(default=60.0)

    def build(self) -> VoyageAICrossEncoder:
        return VoyageAICrossEncoder(
            model=self.model,
            api_key=self.api_key.get_secret_value() if self.api_key else None,
            base_url=self.base_url,
            timeout=self.timeout,
        )


class VoyageAICrossEncoder(AbstractCrossEncoder):
    def __init__(
        self,
        model: str = 'rerank-2.5',
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not api_key and not (api_key := os.getenv('VOYAGE_API_KEY', None)):
            raise ValueError(
                'VoyageAI API key must be provided either via config',
                " or 'VOYAGE_API_KEY' env variable.",
            )

        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.model = model

        self._async_client: AsyncClient | None = None

    @property
    def config(self) -> VoyageAICrossEncoderConfig:
        return VoyageAICrossEncoderConfig(
            model=self.model,
            api_key=SecretStr(self.api_key),
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def __get_async_client(self) -> AsyncClient:
        if self._async_client:
            return self._async_client

        self._async_client = AsyncClient(
            api_key=self.api_key, timeout=self.timeout, base_url=self.base_url
        )
        return self._async_client

    async def _rank_internal(
        self, query: str, texts: Iterable[str]
    ) -> list[tuple[tuple[str, str], float]]:
        """Internal rank method without telemetry, used by predict."""
        texts_list = list(texts)
        ranks = await self.__get_async_client().rerank(
            query, texts_list, model=self.model
        )

        return [
            (
                (query, text),
                score.relevance_score,
            )
            for text, score in zip(texts_list, ranks.results)
        ]

    @tiny_trace('rank')
    async def rank(
        self, query: str, texts: Iterable[str]
    ) -> list[tuple[tuple[str, str], float]]:
        texts_list = list(texts)
        ranks = await self.__get_async_client().rerank(
            query, texts_list, model=self.model
        )

        result = [
            (
                (query, text),
                score.relevance_score,
            )
            for text, score in zip(texts_list, ranks.results)
        ]

        set_cross_encoder_telemetry_attributes(
            self.config,
            query=query,
            texts=texts_list,
            result=result,
        )

        return result

    @tiny_trace('predict')
    async def predict(
        self, pairs: Iterable[tuple[str, str]]
    ) -> list[tuple[tuple[str, str], float]]:
        pairs_list = list(pairs)
        query_text_map: defaultdict[str, list[str]] = defaultdict(list)

        for q, t in pairs_list:
            query_text_map[q].append(t)

        tasks = [
            self._rank_internal(query, text) for query, text in query_text_map.items()
        ]

        results = await run_in_semaphore(*tasks)
        result = [item for sublist in results for item in sublist]

        set_cross_encoder_telemetry_attributes(
            self.config,
            pairs=pairs_list,
            result=result,
        )

        return result

from __future__ import annotations

import os
from typing import Literal

from mistralai import Mistral
from pydantic import Field
from pydantic import SecretStr

from tinygent.core.datamodels.embedder import AbstractEmbedder
from tinygent.core.datamodels.embedder import AbstractEmbedderConfig
from tinygent.core.telemetry.decorators import tiny_trace
from tinygent.core.telemetry.utils import set_embedder_telemetry_attributes

_SUPPORTED_MODELS: dict[str, int] = {
    'mistral-embed': 1024,
    'codestral-embed': 1536,
}


class MistralAIEmbedderConfig(AbstractEmbedderConfig['MistralAIEmbedder']):
    type: Literal['mistralai'] = Field(default='mistralai', frozen=True)

    model: str = Field(default='mistral-embed')

    api_key: SecretStr | None = Field(
        default_factory=lambda: (
            SecretStr(os.environ['MISTRALAI_API_KEY'])
            if 'MISTRALAI_API_KEY' in os.environ
            else None
        ),
    )

    timeout: float = Field(default=60.0)

    def build(self) -> MistralAIEmbedder:
        return MistralAIEmbedder(
            model=self.model,
            api_key=self.api_key.get_secret_value() if self.api_key else None,
            timeout=self.timeout,
        )


class MistralAIEmbedder(AbstractEmbedder):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not api_key and not (api_key := os.getenv('MISTRALAI_API_KEY', None)):
            raise ValueError(
                'MistralAI API key must be provided either via config'
                "or 'MISTRALAI_API_KEY' env variable."
            )

        if model not in _SUPPORTED_MODELS:
            raise ValueError(
                f'Provided model name: {model} not in supported model names: {", ".join(_SUPPORTED_MODELS.keys())}'
            )

        self._client: Mistral | None = None
        self._model = model

        self.timeout = timeout
        self.api_key = api_key

    @property
    def model(self) -> str:
        return self._model

    @property
    def config(self) -> MistralAIEmbedderConfig:
        return MistralAIEmbedderConfig(
            model=self.model,
            api_key=SecretStr(self.api_key),
            timeout=self.timeout,
        )

    @property
    def embedding_dim(self) -> int:
        v = _SUPPORTED_MODELS.get(self.model)
        if not v:
            raise ValueError(
                f'Provided model name: {self.model} not in supported model names: {", ".join(_SUPPORTED_MODELS.keys())}'
            )
        return v

    def __get_client(self) -> Mistral:
        if self._client:
            return self._client

        self._client = Mistral(api_key=self.api_key, timeout_ms=int(self.timeout) * 1000)
        return self._client

    @tiny_trace('embed')
    def embed(self, query: str) -> list[float]:
        res = self.__get_client().embeddings.create(
            model=self.model,
            inputs=[query],
            timeout_ms=int(self.timeout) * 1000,
        )

        embedding = res.data[0].embedding
        if not embedding:
            raise ValueError(f'Error while creating embeddings for query {query}')

        set_embedder_telemetry_attributes(
            self.config,
            query,
            embedding_dim=self.embedding_dim,
            result_len=len(embedding),
        )
        return embedding

    @tiny_trace('embed_batch')
    def embed_batch(self, queries: list[str]) -> list[list[float]]:
        res = self.__get_client().embeddings.create(
            model=self.model,
            inputs=queries,
            timeout_ms=int(self.timeout) * 1000,
        )

        embeddings = []
        for q, r in zip(queries, res.data, strict=True):
            if (e := r.embedding) is None:
                raise ValueError(f'Error while creating embeddings for query {q}')
            embeddings.append(e)

        set_embedder_telemetry_attributes(
            self.config,
            queries,
            embedding_dim=self.embedding_dim,
            result_len=len(embeddings),
        )
        return embeddings

    @tiny_trace('aembed')
    async def aembed(self, query: str) -> list[float]:
        res = await self.__get_client().embeddings.create_async(
            model=self.model,
            inputs=[query],
            timeout_ms=int(self.timeout) * 1000,
        )

        embedding = res.data[0].embedding
        if not embedding:
            raise ValueError(f'Error while creating embeddings for query {query}')

        set_embedder_telemetry_attributes(
            self.config,
            query,
            embedding_dim=self.embedding_dim,
            result_len=len(embedding),
        )
        return embedding

    @tiny_trace('aembed_batch')
    async def aembed_batch(self, queries: list[str]) -> list[list[float]]:
        res = await self.__get_client().embeddings.create_async(
            model=self.model,
            inputs=queries,
            timeout_ms=int(self.timeout) * 1000,
        )

        embeddings = []
        for q, r in zip(queries, res.data, strict=True):
            if (e := r.embedding) is None:
                raise ValueError(f'Error while creating embeddings for query {q}')
            embeddings.append(e)

        set_embedder_telemetry_attributes(
            self.config,
            queries,
            embedding_dim=self.embedding_dim,
            result_len=len(embeddings),
        )
        return embeddings

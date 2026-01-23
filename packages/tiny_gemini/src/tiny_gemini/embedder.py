from __future__ import annotations

import os
from typing import Literal

from google.genai.client import AsyncClient
from google.genai.client import Client
from pydantic import Field
from pydantic import SecretStr

from tinygent.core.datamodels.embedder import AbstractEmbedder
from tinygent.core.datamodels.embedder import AbstractEmbedderConfig
from tinygent.core.telemetry.decorators import tiny_trace
from tinygent.core.telemetry.utils import set_embedder_telemetry_attributes

_SUPPORTED_MODELS: dict[str, int] = {
    'gemini-embedding-001': 3072,
}


class GeminiEmbedderConfig(AbstractEmbedderConfig['GeminiEmbedder']):
    type: Literal['gemini'] = Field(default='gemini', frozen=True)

    model: str = Field(default='gemini-embedding-001')

    api_key: SecretStr | None = Field(
        default_factory=lambda: (
            SecretStr(os.environ['GEMINI_API_KEY'])
            if 'GEMINI_API_KEY' in os.environ
            else None
        ),
    )

    def build(self) -> GeminiEmbedder:
        return GeminiEmbedder(
            model=self.model,
            api_key=self.api_key.get_secret_value() if self.api_key else None,
        )


class GeminiEmbedder(AbstractEmbedder):
    def __init__(
        self,
        model: str = 'gemini-2.5-flash',
        api_key: str | None = None,
    ) -> None:
        if not api_key and not (api_key := os.getenv('GEMINI_API_KEY')):
            raise ValueError(
                'Gemini API key must be provided either via config',
                " or 'GEMINI_API_KEY' env variable.",
            )

        if model not in _SUPPORTED_MODELS:
            raise ValueError(
                f'Provided model name: {model} not in supported model names: {", ".join(_SUPPORTED_MODELS.keys())}'
            )

        self._sync_client: Client | None = Client(api_key=api_key)
        self._model = model

        self.api_key = api_key

    @property
    def model(self) -> str:
        return self._model

    @property
    def config(self) -> GeminiEmbedderConfig:
        return GeminiEmbedderConfig(
            model=self.model,
            api_key=SecretStr(self.api_key),
        )

    @property
    def embedding_dim(self) -> int:
        v = _SUPPORTED_MODELS.get(self.model)
        if not v:
            raise ValueError(
                f'Provided model name: {self.model} not in supported model names: {", ".join(_SUPPORTED_MODELS.keys())}'
            )
        return v

    def __get_sync_client(self) -> Client:
        if self._sync_client:
            return self._sync_client

        self._sync_client = Client(api_key=self.api_key)
        return self._sync_client

    def __get_async_client(self) -> AsyncClient:
        cli = self.__get_sync_client()
        return cli.aio

    @tiny_trace('embed')
    def embed(self, query: str) -> list[float]:
        res = self.__get_sync_client().models.embed_content(
            model=self.model, contents=query
        )
        embedding = res.embeddings[0].values if res.embeddings else None
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
        res = self.__get_sync_client().models.embed_content(
            model=self.model,
            contents=queries,  # type: ignore[arg-type]
        )

        embeddings = []
        for q, r in zip(queries, res.embeddings or [], strict=True):
            if (e := r.values) is None:
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
        res = await self.__get_async_client().models.embed_content(
            model=self.model, contents=query
        )
        embedding = res.embeddings[0].values if res.embeddings else None
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
        res = await self.__get_async_client().models.embed_content(
            model=self.model,
            contents=queries,  # type: ignore[arg-type]
        )

        embeddings = []
        for q, r in zip(queries, res.embeddings or [], strict=True):
            if (e := r.values) is None:
                raise ValueError(f'Error while creating embeddings for query {q}')
            embeddings.append(e)

        set_embedder_telemetry_attributes(
            self.config,
            queries,
            embedding_dim=self.embedding_dim,
            result_len=len(embeddings),
        )
        return embeddings

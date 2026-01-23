from __future__ import annotations

import os
from typing import Literal

from openai import AsyncOpenAI
from openai import OpenAI
from pydantic import Field
from pydantic import SecretStr

from tinygent.core.datamodels.embedder import AbstractEmbedder
from tinygent.core.datamodels.embedder import AbstractEmbedderConfig
from tinygent.core.telemetry.decorators import tiny_trace
from tinygent.core.telemetry.utils import set_embedder_telemetry_attributes

# all supported models with its output embeddings size
_SUPPORTED_MODELS: dict[str, int] = {
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
    'text-embedding-ada-002': 1536,
}


class OpenAIEmbedderConfig(AbstractEmbedderConfig['OpenAIEmbedder']):
    type: Literal['openai'] = Field(default='openai', frozen=True)

    model: str = Field(default='text-embedding-3-small')

    api_key: SecretStr | None = Field(
        default_factory=lambda: (
            SecretStr(os.environ['OPENAI_API_KEY'])
            if 'OPENAI_API_KEY' in os.environ
            else None
        ),
    )

    base_url: str | None = Field(default=None)

    timeout: float = Field(default=60.0)

    def build(self) -> OpenAIEmbedder:
        return OpenAIEmbedder(
            model=self.model,
            api_key=self.api_key.get_secret_value() if self.api_key else None,
            base_url=self.base_url,
            timeout=self.timeout,
        )


class OpenAIEmbedder(AbstractEmbedder):
    def __init__(
        self,
        model: str = 'text-embedding-3-small',
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not api_key and not (api_key := os.getenv('OPENAI_API_KEY', None)):
            raise ValueError(
                'OpenAI API key must be provided either via config',
                " or 'OPENAI_API_KEY' env variable.",
            )

        if model not in _SUPPORTED_MODELS:
            raise ValueError(
                f'Provided model name: {model} not in supported model names: {", ".join(_SUPPORTED_MODELS.keys())}'
            )

        self.api_key = api_key
        self.base_url = base_url
        self._model = model
        self._timeout = timeout

        self.__sync_client: OpenAI | None = None
        self.__async_client: AsyncOpenAI | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def config(self) -> OpenAIEmbedderConfig:
        return OpenAIEmbedderConfig(
            model=self.model,
            api_key=SecretStr(self.api_key),
            base_url=self.base_url,
            timeout=self._timeout,
        )

    @property
    def embedding_dim(self) -> int:
        v = _SUPPORTED_MODELS.get(self.model)
        if not v:
            raise ValueError(
                f'Provided model name: {self.model} not in supported model names: {", ".join(_SUPPORTED_MODELS.keys())}'
            )
        return v

    def __get_sync_client(self) -> OpenAI:
        if self.__sync_client:
            return self.__sync_client

        self.__sync_client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self._timeout
        )
        return self.__sync_client

    def __get_async_client(self) -> AsyncOpenAI:
        if self.__async_client:
            return self.__async_client

        self.__async_client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self._timeout
        )
        return self.__async_client

    @tiny_trace('embed')
    def embed(self, query: str) -> list[float]:
        res = self.__get_sync_client().embeddings.create(
            input=query,
            model=self.model,
        )
        embedding = res.data[0].embedding

        set_embedder_telemetry_attributes(
            self.config,
            query,
            embedding_dim=self.embedding_dim,
            result_len=len(embedding),
        )
        return embedding

    @tiny_trace('embed_batch')
    def embed_batch(self, queries: list[str]) -> list[list[float]]:
        res = self.__get_sync_client().embeddings.create(
            input=queries,
            model=self.model,
        )
        embeddings = [emb.embedding for emb in res.data]

        set_embedder_telemetry_attributes(
            self.config,
            queries,
            embedding_dim=self.embedding_dim,
            result_len=len(embeddings),
        )
        return embeddings

    @tiny_trace('aembed')
    async def aembed(self, query: str) -> list[float]:
        res = await self.__get_async_client().embeddings.create(
            input=query,
            model=self.model,
        )
        embedding = res.data[0].embedding

        set_embedder_telemetry_attributes(
            self.config,
            query,
            embedding_dim=self.embedding_dim,
            result_len=len(embedding),
        )
        return embedding

    @tiny_trace('aembed_batch')
    async def aembed_batch(self, queries: list[str]) -> list[list[float]]:
        res = await self.__get_async_client().embeddings.create(
            input=queries,
            model=self.model,
        )
        embeddings = [emb.embedding for emb in res.data]

        set_embedder_telemetry_attributes(
            self.config,
            queries,
            embedding_dim=self.embedding_dim,
            result_len=len(embeddings),
        )
        return embeddings

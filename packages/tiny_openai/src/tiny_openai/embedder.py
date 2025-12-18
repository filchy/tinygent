from __future__ import annotations

import os
from typing import Literal

from openai import AsyncOpenAI
from openai import OpenAI

from tinygent.datamodels.embedder import AbstractEmbedder
from tinygent.datamodels.embedder import AbstractEmbedderConfig

# all supported models with its output embeddings size
_SUPPORTED_MODELS: dict[str, int] = {
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
    'text-embedding-ada-002': 1536,
}


class OpenAIEmbedderConfig(AbstractEmbedderConfig['OpenAIEmbedder']):
    type: Literal['openai'] = 'openai'

    model: str = 'text-embedding-3-small'

    api_key: str | None = os.getenv('OPENAI_API_KEY', None)

    base_url: str | None = None

    def build(self) -> OpenAIEmbedder:
        return OpenAIEmbedder(
            model_name=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
        )


class OpenAIEmbedder(AbstractEmbedder):
    def __init__(
        self,
        model_name: str = 'text-embedding-3-small',
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if not api_key and not (api_key := os.getenv('OPENAI_API_KEY', None)):
            raise ValueError(
                'OpenAI API key must be provided either via config',
                " or 'OPENAI_API_KEY' env variable.",
            )

        if model_name not in _SUPPORTED_MODELS:
            raise ValueError(
                f'Provided model name: {model_name} not in supported model names: {', '.join(_SUPPORTED_MODELS.keys())}'
            )

        self.api_key = api_key
        self.base_url = base_url
        self._model_name = model_name

        self.__sync_client: OpenAI | None = None
        self.__async_client: AsyncOpenAI | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        v = _SUPPORTED_MODELS.get(self.model_name)
        if not v:
            raise ValueError(
                f'Provided model name: {self.model_name} not in supported model names: {', '.join(_SUPPORTED_MODELS.keys())}'
            )
        return v

    def __get_sync_client(self) -> OpenAI:
        if self.__sync_client:
            return self.__sync_client

        self.__sync_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self.__sync_client

    def __get_async_client(self) -> AsyncOpenAI:
        if self.__async_client:
            return self.__async_client

        self.__async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self.__async_client

    def embed(self, query: str) -> list[float]:
        res = self.__get_sync_client().embeddings.create(
            input=query,
            model=self.model_name,
        )
        return res.data[0].embedding

    def embed_batch(self, queries: list[str]) -> list[list[float]]:
        res = self.__get_sync_client().embeddings.create(
            input=queries,
            model=self.model_name,
        )
        return [emb.embedding for emb in res.data]

    async def aembed(self, query: str) -> list[float]:
        res = await self.__get_async_client().embeddings.create(
            input=query,
            model=self.model_name,
        )
        return res.data[0].embedding

    async def aembed_batch(self, queries: list[str]) -> list[list[float]]:
        res = await self.__get_async_client().embeddings.create(
            input=queries,
            model=self.model_name,
        )
        return [emb.embedding for emb in res.data]

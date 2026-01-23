import os
from typing import Literal

from pydantic import Field
from pydantic import SecretStr
from voyageai.client import Client
from voyageai.client_async import AsyncClient

from tinygent.core.datamodels.embedder import AbstractEmbedder
from tinygent.core.datamodels.embedder import AbstractEmbedderConfig
from tinygent.core.telemetry.decorators import tiny_trace
from tinygent.core.telemetry.utils import set_embedder_telemetry_attributes

_SUPPORTED_MODELS: dict[str, int] = {
    'voyage-3-large': 1024,
    'voyage-3.5': 1024,
    'voyage-3.5-lite': 1024,
    'voyage-code-3': 1024,
    'voyage-finance-2': 1024,
    'voyage-law-2': 1024,
    'voyage-code-2': 1536,
}


class VoyageAIEmbedderConfig(AbstractEmbedderConfig['VoyageAIEmbedder']):
    type: Literal['voyageai'] = Field(default='voyageai', frozen=True)

    model: str = Field(default='voyage-3-large')

    api_key: SecretStr | None = Field(
        default_factory=lambda: (
            SecretStr(os.environ['VOYAGE_API_KEY'])
            if 'VOYAGE_API_KEY' in os.environ
            else None
        ),
    )

    base_url: str | None = Field(default=None)

    timeout: float = Field(default=60.0)

    def build(self) -> 'VoyageAIEmbedder':
        return VoyageAIEmbedder(
            model=self.model,
            api_key=self.api_key.get_secret_value() if self.api_key else None,
            base_url=self.base_url,
            timeout=self.timeout,
        )


class VoyageAIEmbedder(AbstractEmbedder):
    def __init__(
        self,
        model: str = 'voyage-3-large',
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not api_key and not (api_key := os.getenv('VOYAGE_API_KEY', None)):
            raise ValueError(
                'VoyageAI API key must be provided either via config',
                " or 'VOYAGE_API_KEY' env variable.",
            )

        if model not in _SUPPORTED_MODELS:
            raise ValueError(
                f'Provided model name: {model} not in supported model names: {", ".join(_SUPPORTED_MODELS.keys())}'
            )

        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self._model = model

        self._sync_client: Client | None = None
        self._async_client: AsyncClient | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def config(self) -> VoyageAIEmbedderConfig:
        return VoyageAIEmbedderConfig(
            model=self.model,
            api_key=SecretStr(self.api_key),
            base_url=self.base_url,
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

    def __get_sync_client(self) -> Client:
        if self._sync_client:
            return self._sync_client

        self._sync_client = Client(
            api_key=self.api_key, timeout=self.timeout, base_url=self.base_url
        )
        return self._sync_client

    def __get_async_client(self) -> AsyncClient:
        if self._async_client:
            return self._async_client

        self._async_client = AsyncClient(
            api_key=self.api_key, timeout=self.timeout, base_url=self.base_url
        )
        return self._async_client

    @tiny_trace('embed')
    def embed(self, query: str) -> list[float]:
        results = self.__get_sync_client().embed(
            texts=[query],
            model=self.model,
        )

        embedding = results.embeddings[0]

        set_embedder_telemetry_attributes(
            self.config,
            query,
            embedding_dim=self.embedding_dim,
            result_len=len(embedding),
        )
        return list(map(float, embedding))

    @tiny_trace('embed_batch')
    def embed_batch(self, queries: list[str]) -> list[list[float]]:
        results = self.__get_sync_client().embed(
            texts=queries,
            model=self.model,
        )

        embeddings = [list(map(float, e)) for e in results.embeddings]

        set_embedder_telemetry_attributes(
            self.config,
            queries,
            embedding_dim=self.embedding_dim,
            result_len=len(embeddings),
        )
        return embeddings

    @tiny_trace('aembed')
    async def aembed(self, query: str) -> list[float]:
        results = await self.__get_async_client().embed(
            texts=[query],
            model=self.model,
        )

        embedding = results.embeddings[0]

        set_embedder_telemetry_attributes(
            self.config,
            query,
            embedding_dim=self.embedding_dim,
            result_len=len(embedding),
        )
        return list(map(float, embedding))

    @tiny_trace('aembed_batch')
    async def aembed_batch(self, queries: list[str]) -> list[list[float]]:
        results = await self.__get_async_client().embed(
            texts=queries,
            model=self.model,
        )

        embeddings = [list(map(float, e)) for e in results.embeddings]

        set_embedder_telemetry_attributes(
            self.config,
            queries,
            embedding_dim=self.embedding_dim,
            result_len=len(embeddings),
        )
        return embeddings

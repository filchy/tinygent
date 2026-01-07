from dataclasses import dataclass

from tiny_graph.driver.base import BaseDriver
from tinygent.datamodels.cross_encoder import AbstractCrossEncoder
from tinygent.datamodels.embedder import AbstractEmbedder
from tinygent.datamodels.llm import AbstractLLM


@dataclass
class TinyGraphClients:
    driver: BaseDriver
    llm: AbstractLLM
    embedder: AbstractEmbedder
    cross_encoder: AbstractCrossEncoder

    @property
    def safe_embed_model(self) -> str:
        return self.embedder.model_name.replace('-', '_')

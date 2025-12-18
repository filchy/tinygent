from dataclasses import dataclass
from tinygent.datamodels.llm import AbstractLLM
from tinygent.datamodels.embedder import AbstractEmbedder

from tiny_graph.driver.base import BaseDriver


@dataclass
class TinyGraphClients:
    driver: BaseDriver
    llm: AbstractLLM
    embedder: AbstractEmbedder

    @property
    def safe_embed_model(self) -> str:
        return self.embedder.model_name.replace('-', '_')

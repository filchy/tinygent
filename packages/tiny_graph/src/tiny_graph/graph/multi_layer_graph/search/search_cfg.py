from enum import Enum
from pydantic import Field
from tinygent.types.base import TinyModel


class EntitySearchMethods(Enum):
    COSINE_SIM = 'cosine_similarity'
    BM_25 = 'bm_25'


class TinyEntitySearchConfig(TinyModel):
    search_methods: list[EntitySearchMethods] = Field(default=[EntitySearchMethods.COSINE_SIM])


class TinySearchConfig(TinyModel):
    limit: int = Field(default=5)

    entity_search: TinyEntitySearchConfig = Field(default_factory=TinyEntitySearchConfig)

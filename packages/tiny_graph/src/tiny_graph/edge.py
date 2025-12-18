from datetime import datetime
from pydantic import Field
from tinygent.types.base import TinyModel

from tiny_graph.helper import generate_uuid
from tiny_graph.helper import get_current_timestamp
from tiny_graph.node import TinyNode


class TinyEdge(TinyModel):
    uuid: str = Field(description='unique edge identifier', default_factory=generate_uuid)

    group_id: str = Field(..., description='subgraph identifier')

    source_node: TinyNode

    target_node: TinyNode

    created_at: datetime = Field(default_factory=get_current_timestamp)

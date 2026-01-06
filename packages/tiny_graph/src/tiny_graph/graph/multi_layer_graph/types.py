from enum import Enum


class NodeType(Enum):
    EVENT = 'Event'
    ENTITY = 'Entity'
    CLUSTER = 'Cluster'


class EdgeType(Enum):
    HAS_MEMBER = 'HasMember'
    MENTIONS = 'Mentions'


class DataType(Enum):
    TEXT = 'text'
    JSON = 'json'
    MESSAGE = 'message'

from enum import Enum


class NodeType(Enum):
    EVENT = 'Event'
    ENTITY = 'Entity'
    CLUSTER = 'Cluster'


class DataType(Enum):
    TEXT = 'text'
    JSON = 'json'
    MESSAGE = 'message'

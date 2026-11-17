from dataclasses import dataclass
from enum import Enum


class NodeType(Enum):
    INPUT = "input"
    BIAS = "bias"
    HIDDEN = "hidden"
    OUTPUT = "output"


@dataclass
class NodeGene:
    id: int
    type: NodeType


@dataclass
class ConnectionGene:
    in_node: int
    out_node: int
    weight: float
    enabled: bool
    innovation: int

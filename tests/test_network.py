import random

import pytest

from simsengine.neat.genes import NodeGene, NodeType
from simsengine.neat.genome import Genome
from simsengine.neat.innovation import InnovationTracker
from simsengine.neat.network import FeedForwardNetwork

XOR_CASES = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]


def test_hand_built_xor_network():
    tracker = InnovationTracker(next_node_id=4)
    g = Genome(2, 1)
    for i in g.input_ids:
        g.nodes[i] = NodeGene(i, NodeType.INPUT)
    g.nodes[g.bias_id] = NodeGene(g.bias_id, NodeType.BIAS)
    g.nodes[3] = NodeGene(3, NodeType.OUTPUT)
    g.nodes[4] = NodeGene(4, NodeType.HIDDEN)
    for a, b, w in [
        (0, 4, 1.0),
        (1, 4, 1.0),
        (2, 4, -1.5),
        (0, 3, 1.0),
        (1, 3, 1.0),
        (2, 3, -0.5),
        (4, 3, -2.0),
    ]:
        g.add_connection(a, b, w, tracker)

    net = FeedForwardNetwork.create(g)
    for x, y in XOR_CASES:
        assert (net.activate(x)[0] > 0.5) == (y == 1)


def test_wrong_number_of_inputs_is_an_error():
    g = Genome.initial(2, 1, InnovationTracker(next_node_id=4), random.Random(0))
    with pytest.raises(ValueError):
        FeedForwardNetwork.create(g).activate([1.0])

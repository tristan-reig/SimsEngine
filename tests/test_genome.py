import random

import pytest

from simsengine.neat.config import NEATConfig
from simsengine.neat.genes import NodeType
from simsengine.neat.genome import Genome
from simsengine.neat.innovation import InnovationTracker


@pytest.fixture
def tracker():
    return InnovationTracker(next_node_id=4)  # 2 entrées + biais + 1 sortie


def make_genome(tracker, seed=0):
    return Genome.initial(2, 1, tracker, random.Random(seed))


def count_hidden(genome):
    return sum(1 for n in genome.nodes.values() if n.type is NodeType.HIDDEN)


def test_initial_genome_connects_every_input_and_bias_to_every_output(tracker):
    g = Genome.initial(3, 2, InnovationTracker(next_node_id=6), random.Random(0))
    assert len(g.nodes) == 3 + 1 + 2
    assert len(g.connections) == (3 + 1) * 2
    assert all(c.enabled for c in g.connections.values())


def test_same_connection_gets_same_innovation_in_two_genomes(tracker):
    a, b = make_genome(tracker, 1), make_genome(tracker, 2)
    assert a.connections.keys() == b.connections.keys() == {0, 1, 2}


def test_add_node_splits_a_connection(tracker):
    g = make_genome(tracker)
    assert g.mutate_add_node(tracker, random.Random(0))
    assert count_hidden(g) == 1
    disabled = [c for c in g.connections.values() if not c.enabled]
    assert len(disabled) == 1
    old = disabled[0]
    hidden = next(n.id for n in g.nodes.values() if n.type is NodeType.HIDDEN)
    into = next(c for c in g.connections.values() if c.out_node == hidden)
    out_of = next(c for c in g.connections.values() if c.in_node == hidden)
    assert (into.in_node, into.weight) == (old.in_node, 1.0)
    assert (out_of.out_node, out_of.weight) == (old.out_node, old.weight)


def test_same_split_in_two_genomes_gives_same_node_and_innovations(tracker):
    a, b = make_genome(tracker, 7), make_genome(tracker, 8)
    a.mutate_add_node(tracker, random.Random(5))
    b.mutate_add_node(tracker, random.Random(5))  # même graine : même connexion coupée
    assert a.nodes.keys() == b.nodes.keys()
    assert a.connections.keys() == b.connections.keys()


def test_many_mutations_never_create_a_cycle(tracker):
    config = NEATConfig()
    rng = random.Random(1)
    g = make_genome(tracker)
    for _ in range(500):
        g.mutate(config, tracker, rng)
    assert count_hidden(g) > 0
    for c in list(g.connections.values()):
        del g.connections[c.innovation]
        assert not g.creates_cycle(c.in_node, c.out_node)
        g.connections[c.innovation] = c


def test_copy_is_independent(tracker):
    g = make_genome(tracker)
    clone = g.copy()
    clone.mutate_weights(NEATConfig(), random.Random(0))
    assert [c.weight for c in g.connections.values()] != [
        c.weight for c in clone.connections.values()
    ]


def test_distance(tracker):
    config = NEATConfig()
    g = make_genome(tracker)
    assert g.distance(g, config) == 0.0
    bigger = g.copy()
    bigger.mutate_add_node(tracker, random.Random(0))
    assert g.distance(bigger, config) == pytest.approx(2 * config.c1)


def test_serialization_roundtrip(tracker):
    g = make_genome(tracker)
    g.mutate_add_node(tracker, random.Random(0))
    g.fitness = 3.5
    assert Genome.from_dict(g.to_dict()).to_dict() == g.to_dict()

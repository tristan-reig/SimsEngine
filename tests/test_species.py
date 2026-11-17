import random
from dataclasses import replace

from simsengine.neat.config import NEATConfig
from simsengine.neat.crossover import crossover
from simsengine.neat.genome import Genome
from simsengine.neat.innovation import InnovationTracker
from simsengine.neat.species import SpeciesSet


def mutated_population(tracker, rng, size=150, mutations=30):
    config = NEATConfig()
    population = []
    for _ in range(size):
        g = Genome.initial(2, 1, tracker, rng)
        for _ in range(mutations):
            g.mutate(config, tracker, rng)
        population.append(g)
    return population


def test_lower_threshold_gives_more_species():
    rng = random.Random(3)
    tracker = InnovationTracker(next_node_id=4)
    population = mutated_population(tracker, rng)
    counts = []
    for threshold in (1.0, 3.0, 6.0):
        species_set = SpeciesSet()
        config = replace(NEATConfig(), compatibility_threshold=threshold)
        species_set.speciate(population, config, rng)
        assert sum(len(s.members) for s in species_set.species.values()) == len(population)
        counts.append(len(species_set.species))
    assert counts[0] > counts[1] > counts[2] >= 1


def test_crossover_takes_structure_from_fitter_parent():
    config = NEATConfig()
    tracker = InnovationTracker(next_node_id=4)
    base = Genome.initial(2, 1, tracker, random.Random(4))
    a, b = base.copy(), base.copy()
    a.mutate_add_node(tracker, random.Random(0))
    b.mutate_add_node(tracker, random.Random(5))
    weights_before = [c.weight for c in a.connections.values()]

    child = crossover(a, b, config, random.Random(4))
    assert child.connections.keys() == a.connections.keys()
    assert child.nodes.keys() == a.nodes.keys()

    child.mutate_weights(config, random.Random(0))
    assert [c.weight for c in a.connections.values()] == weights_before

import pytest

from simsengine.neat.config import NEATConfig
from simsengine.neat.network import FeedForwardNetwork
from simsengine.neat.population import Population

XOR_CASES = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]


def evaluate(genomes):
    fitnesses = []
    for g in genomes:
        net = FeedForwardNetwork.create(g)
        error = sum(abs(net.activate(x)[0] - y) for x, y in XOR_CASES)
        fitnesses.append((4 - error) ** 2)
    return fitnesses


def solves_xor(genome):
    net = FeedForwardNetwork.create(genome)
    return all((net.activate(x)[0] > 0.5) == (y == 1) for x, y in XOR_CASES)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_neat_solves_xor(seed):
    pop = Population(NEATConfig(), 2, 1, seed=seed)
    best = pop.run(evaluate, max_generations=300, stop_when=solves_xor, verbose=False)
    assert solves_xor(best)


def test_evolution_is_reproducible():
    histories = []
    for _ in range(2):
        pop = Population(NEATConfig(), 2, 1, seed=0)
        pop.run(evaluate, max_generations=10, verbose=False)
        histories.append(pop.history)
    assert histories[0] == histories[1]


def test_adaptive_threshold_creates_several_species():
    pop = Population(NEATConfig(), 2, 1, seed=0)
    pop.run(evaluate, max_generations=10, verbose=False)
    assert max(h["species"] for h in pop.history) > 1

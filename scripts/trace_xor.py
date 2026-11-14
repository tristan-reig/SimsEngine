"""Trace des 12 premières générations de XOR."""

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


Population(NEATConfig(), 2, 1, seed=0).run(evaluate, 12)

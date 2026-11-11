import random

from simsengine.neat.config import NEATConfig
from simsengine.neat.genome import Genome
from simsengine.neat.innovation import InnovationTracker
from simsengine.neat.network import FeedForwardNetwork

XOR_INPUTS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def outputs(genome):
    net = FeedForwardNetwork.create(genome)
    return [round(net.activate(x)[0], 3) for x in XOR_INPUTS]


config = NEATConfig()
rng = random.Random(1)
tracker = InnovationTracker(next_node_id=4)

# Ajout d'un nœud
g = Genome.initial(2, 1, tracker, rng)
print("Avant ajout de nœud")
print(g.describe())
print("  sorties :", outputs(g))

g.mutate_add_node(tracker, rng)
print("\nAprès ajout de nœud")
print(g.describe())
print("  sorties :", outputs(g))

# Ajout d'une connexion
g.mutate_add_connection(config, tracker, rng)
print("\nAprès ajout de connexion")
print(g.describe())

# Deux génomes qui font la même coupe obtiennent les mêmes numéros
a = Genome.initial(2, 1, tracker, random.Random(7))
b = Genome.initial(2, 1, tracker, random.Random(8))
a.mutate_add_node(tracker, random.Random(5))
b.mutate_add_node(tracker, random.Random(5))
print("\nMême coupe dans A et B -> mêmes innovations :", a.connections.keys() == b.connections.keys())

# 4. Test d'endurance
for _ in range(500):
    g.mutate(config, tracker, rng)
n_hidden = sum(1 for n in g.nodes.values() if n.type.value == "hidden")
print(f"\nAprès 500 mutations : {n_hidden} nœuds cachés, {len(g.connections)} connexions")
print("  le réseau se calcule toujours :", outputs(g))

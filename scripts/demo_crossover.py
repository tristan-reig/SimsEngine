import random

from simsengine.neat.config import NEATConfig
from simsengine.neat.crossover import crossover
from simsengine.neat.genome import Genome
from simsengine.neat.innovation import InnovationTracker

config = NEATConfig()
rng = random.Random(4)
tracker = InnovationTracker(next_node_id=4)

# Deux parents issus du même génome, qui ont évolué différemment
base = Genome.initial(2, 1, tracker, rng)

a = base.copy()
a.mutate_add_node(tracker, random.Random(0))
a.mutate_weights(config, rng)

b = base.copy()
b.mutate_add_node(tracker, random.Random(5))
b.mutate_add_connection(config, tracker, rng)
b.mutate_weights(config, rng)

print("Parent A (le meilleur)")
print(a.describe())
print("\nParent B")
print(b.describe())

child = crossover(a, b, config, rng)
print("\nEnfant")
print(child.describe())

print("\nL'enfant a la structure de A :", child.connections.keys() == a.connections.keys())

origines = []
for innov, gene in child.connections.items():
    if innov not in b.connections:
        origines.append(f"{innov}:A seul")
    elif gene.weight == a.connections[innov].weight:
        origines.append(f"{innov}:A")
    else:
        origines.append(f"{innov}:B")
print("Origine de chaque poids :", ", ".join(origines))

poids_a_avant = [c.weight for c in a.connections.values()]
child.mutate_weights(config, rng)
print("Parent A intact après mutation de l'enfant :",
      [c.weight for c in a.connections.values()] == poids_a_avant)

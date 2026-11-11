import random

from simsengine.neat.genes import NodeGene, NodeType
from simsengine.neat.genome import Genome
from simsengine.neat.innovation import InnovationTracker
from simsengine.neat.network import FeedForwardNetwork

XOR_CASES = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]


def show(net):
    for x, y in XOR_CASES:
        out = net.activate(x)[0]
        print(f"  {x} -> {out:.3f}   (attendu {y})")


rng = random.Random(0)
tracker = InnovationTracker(next_node_id=4)

# Génome initial aléatoire
g = Genome.initial(2, 1, tracker, rng)
print("Génome initial (poids aléatoires)")
show(FeedForwardNetwork.create(g))

# Réseau XOR construit à la main, avec un noeud caché
xor = Genome(2, 1)
for i in xor.input_ids:
    xor.nodes[i] = NodeGene(i, NodeType.INPUT)
xor.nodes[xor.bias_id] = NodeGene(xor.bias_id, NodeType.BIAS)
xor.nodes[3] = NodeGene(3, NodeType.OUTPUT)
xor.nodes[4] = NodeGene(4, NodeType.HIDDEN)

for a, b, w in [
    (0, 4, 1.0), (1, 4, 1.0), (2, 4, -1.5),
    (0, 3, 1.0), (1, 3, 1.0), (2, 3, -0.5), (4, 3, -2.0),
]:
    xor.add_connection(a, b, w, tracker)

print("\nRéseau XOR construit à la main")
print(xor.describe())
show(FeedForwardNetwork.create(xor))

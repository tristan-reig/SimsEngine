import random

from simsengine.neat.genome import Genome
from simsengine.neat.innovation import InnovationTracker

N_INPUTS, N_OUTPUTS = 2, 1

rng = random.Random(0)
tracker = InnovationTracker(next_node_id=N_INPUTS + 1 + N_OUTPUTS)
a = Genome.initial(N_INPUTS, N_OUTPUTS, tracker, rng)
b = Genome.initial(N_INPUTS, N_OUTPUTS, tracker, rng)
attendu = (N_INPUTS + 1) * N_OUTPUTS


print("Génome A")
print(a.describe())
print("\nGénome B")
print(b.describe())
print("\nMêmes numéros d'innovation :", a.connections.keys() == b.connections.keys())
print(f"Nombre de connexions : {len(a.connections)} (attendu {attendu})")

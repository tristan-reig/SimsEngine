import random
from dataclasses import replace

from simsengine.neat.config import NEATConfig
from simsengine.neat.genome import Genome
from simsengine.neat.innovation import InnovationTracker
from simsengine.neat.species import SpeciesSet

config = NEATConfig()
rng = random.Random(3)
tracker = InnovationTracker(next_node_id=4)

# Distance sur quelques cas simples
g = Genome.initial(2, 1, tracker, rng)

poids = g.copy()
poids.mutate_weights(config, rng)

structure = g.copy()
structure.mutate_add_node(tracker, rng)

print("Distances depuis un génome g")
print(f"  g lui-même                : {g.distance(g, config):.3f}")
print(f"  g avec poids mutés        : {g.distance(poids, config):.3f}   (seulement c3 * W)")
print(f"  g avec un nœud en plus    : {g.distance(structure, config):.3f}   (2 gènes en excès)")

# Test sur une population de 150 génomes qui ont subi 30 mutations chacun
population = []
for _ in range(150):
    genome = Genome.initial(2, 1, tracker, rng)
    for _ in range(30):
        genome.mutate(config, tracker, rng)
    population.append(genome)

print("\nSpéciation de 150 génomes mutés")
for seuil in [1.0, 3.0, 6.0]:
    species_set = SpeciesSet()
    species_set.speciate(population, replace(config, compatibility_threshold=seuil), rng)
    tailles = sorted((len(s.members) for s in species_set.species.values()), reverse=True)
    print(f"  seuil {seuil:3.1f} : {len(tailles):3d} espèces, tailles {tailles[:10]}{' ...' if len(tailles) > 10 else ''}")

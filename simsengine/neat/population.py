import math
import random

from .crossover import crossover
from .genome import Genome
from .innovation import InnovationTracker
from .species import SpeciesSet


class Population:
    """Fait évoluer la population."""

    def __init__(self, config, n_inputs, n_outputs, seed=None):
        self.config = config
        self.rng = random.Random(seed)
        self.tracker = InnovationTracker(next_node_id=n_inputs + 1 + n_outputs)
        self.genomes = [
            Genome.initial(n_inputs, n_outputs, self.tracker, self.rng)
            for _ in range(config.population_size)
        ]
        self.species_set = SpeciesSet()
        self.species_set.speciate(self.genomes, config, self.rng)
        self.generation = 0

    def run(self, evaluate, max_generations, stop_when=None, verbose=True):
        """Boucle d'évolution."""
        for _ in range(max_generations):
            # Évaluation
            for g, f in zip(self.genomes, evaluate(self.genomes)):
                g.fitness = f
            best = max(self.genomes, key=lambda g: g.fitness)

            if verbose:
                mean = sum(g.fitness for g in self.genomes) / len(self.genomes)
                n_hidden = sum(1 for n in best.nodes.values() if n.type.value == "hidden")
                print(f"Gén {self.generation:3d} | meilleure {best.fitness:7.3f} | "
                      f"moyenne {mean:7.3f} | espèces {len(self.species_set.species):3d} | "
                      f"cachés du meilleur {n_hidden}")

            if stop_when is not None and stop_when(best):
                return best

            # Reproduction et spéciation
            self.genomes = self._reproduce()
            self.species_set.speciate(self.genomes, self.config, self.rng)
            self.generation += 1

        return best

    def _reproduce(self):
        cfg = self.config
        species = list(self.species_set.species.values())

        # Rafraichissement de de la stagnation
        for s in species:
            s.update_stagnation()
        species.sort(key=lambda s: s.best_fitness, reverse=True)
        survivors = [
            s for i, s in enumerate(species)
            if i < cfg.species_elitism or s.stagnation < cfg.stagnation_limit
        ]

        # Partage de fitness
        lowest = min(m.fitness for s in survivors for m in s.members)
        scores = [sum(m.fitness - lowest for m in s.members) / len(s.members) for s in survivors]
        counts = allocate(scores, cfg.population_size)

        # Reproduction à l'intérieur de chaque espèce
        children = []
        for s, n in zip(survivors, counts):
            children.extend(self._breed(s, n))
        return children

    def _breed(self, species, n_children):
        cfg, rng = self.config, self.rng
        if n_children == 0:
            return []
        members = sorted(species.members, key=lambda g: g.fitness, reverse=True)
        children = []

        # Le meilleur membre d'une espèce rejoint l'élite
        if len(members) > cfg.elitism_min_species_size:
            children.append(members[0].copy())

        # Seuls les meilleurs de l'espèce peuvent se reproduire
        n_parents = max(1, math.ceil(cfg.survival_threshold * len(members)))
        parents = members[:n_parents]

        while len(children) < n_children:
            p1 = rng.choice(parents)
            if rng.random() < cfg.p_crossover:
                p2 = rng.choice(parents)
                fitter, other = (p1, p2) if p1.fitness >= p2.fitness else (p2, p1)
                child = crossover(fitter, other, cfg, rng)
            else:
                child = p1.copy()
            child.mutate(cfg, self.tracker, rng)
            children.append(child)
        return children


def allocate(scores, total):
    """Répartit `total` places entières proportionnellement aux scores.

    Exemple : scores [3, 1], total 10 -> [8, 2] (7.5 et 2.5 arrondis
    en donnant les places restantes aux plus grandes parties décimales).
    """
    s = sum(scores)
    if s <= 0:
        scores, s = [1.0] * len(scores), float(len(scores))
    raw = [total * x / s for x in scores]
    counts = [int(r) for r in raw]
    remaining = total - sum(counts)
    by_decimal = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in by_decimal[:remaining]:
        counts[i] += 1
    return counts

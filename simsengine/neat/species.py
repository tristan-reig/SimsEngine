class Species:
    """Un groupe de génomes proches les uns des autres."""

    def __init__(self, key, representative):
        self.key = key
        self.representative = representative  # génome de référence
        self.members = []
        self.best_fitness = float("-inf")
        self.stagnation = 0  # générations écoulées depuis le dernier progrès

    def update_stagnation(self):
        best = max(m.fitness for m in self.members)
        if best > self.best_fitness:
            self.stagnation = 0
        else:
            self.stagnation += 1

class SpeciesSet:
    """L'ensemble des espèces, mis à jour à chaque génération."""

    def __init__(self) -> None:
        self.species = {}
        self._next_key = 0

    def speciate(self, genomes, config, rng):
        """Range chaque génome dans l'espèce correspondante"""
        for s in self.species.values():
            s.members = []

        for g in genomes:
            for s in self.species.values():
                if g.distance(s.representative, config) < config.compatibility_threshold:
                    s.members.append(g)
                    break
            else:  # aucune espèce compatible : on en crée une
                s = Species(self._next_key, g)
                s.members.append(g)
                self.species[s.key] = s
                self._next_key += 1

        self.species = {k: s for k, s in self.species.items() if s.members}
        for s in self.species.values():
            s.representative = rng.choice(s.members)
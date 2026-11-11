from dataclasses import replace

from .genome import Genome


def crossover(fitter, other, config, rng):
    """Crée un enfant à partir de deux parents."""
    child = Genome(fitter.n_inputs, fitter.n_outputs)
    child.nodes = {k: replace(n) for k, n in fitter.nodes.items()}

    for innov, gene1 in fitter.connections.items():
        gene2 = other.connections.get(innov)
        if gene2 is None:
            gene = replace(gene1)  # gène du meilleur parent
        else:
            gene = replace(gene1 if rng.random() < 0.5 else gene2) # en cas de gene commun, on choisit au hasard
            if not gene1.enabled or not gene2.enabled:
                gene.enabled = rng.random() >= config.p_disable_inherited
        child.connections[innov] = gene

    return child

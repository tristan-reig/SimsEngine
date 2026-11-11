import random
from dataclasses import replace

from .genes import ConnectionGene, NodeGene, NodeType

class Genome:
    """Un génome NEAT : une liste de nœuds et une liste de connexions."""

    def __init__(self, n_inputs, n_outputs) -> None:
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.nodes = {}
        self.connections = {}
        self.fitness = None

    @property
    def input_ids(self):
        return list(range(self.n_inputs))

    @property
    def bias_id(self):
        return self.n_inputs

    @property
    def output_ids(self):
        start = self.n_inputs + 1
        return list(range(start, start + self.n_outputs))

    @classmethod
    def initial(cls, n_inputs, n_outputs, tracker, rng):
        """Génome minimal : chaque entrée (et le biais) reliée à chaque sortie."""
        g = cls(n_inputs, n_outputs)
        for i in g.input_ids:
            g.nodes[i] = NodeGene(i, NodeType.INPUT)
        g.nodes[g.bias_id] = NodeGene(g.bias_id, NodeType.BIAS)
        for o in g.output_ids:
            g.nodes[o] = NodeGene(o, NodeType.OUTPUT)

        for src in g.input_ids + [g.bias_id]:
            for dst in g.output_ids:
                g.add_connection(src, dst, rng.gauss(0.0, 1.0), tracker)
        return g

    def copy(self):
        """Copie indépendante"""
        g = Genome(self.n_inputs, self.n_outputs)
        g.nodes = {k: replace(n) for k, n in self.nodes.items()}
        g.connections = {k: replace(c) for k, c in self.connections.items()}
        return g
    
    def add_connection(self, in_node, out_node, weight, tracker):
        innov = tracker.connection_innovation(in_node, out_node)
        self.connections[innov] = ConnectionGene(in_node, out_node, weight, True, innov)

    def mutate(self, config, tracker, rng):
        """Applique chaque mutation avec sa probabilité."""
        if rng.random() < config.p_add_node:
            self.mutate_add_node(tracker, rng)
        if rng.random() < config.p_add_connection:
            self.mutate_add_connection(config, tracker, rng)
        if rng.random() < config.p_mutate_weights:
            self.mutate_weights(config, rng)

    def mutate_weights(self, config, rng):
        for c in self.connections.values():
            if rng.random() < config.p_weight_perturb:
                c.weight += rng.gauss(0.0, config.weight_perturb_stdev)
            else:
                c.weight = rng.gauss(0.0, config.weight_init_stdev)
            c.weight = max(-config.weight_max, min(config.weight_max, c.weight))

    def mutate_add_node(self, tracker, rng):
        """Coupe une connexion A -> B en A -> N -> B."""
        enabled = [c for c in self.connections.values() if c.enabled]
        if not enabled:
            return False
        old = rng.choice(enabled)

        new_id = tracker.split_node_id(old.innovation)
        if new_id in self.nodes:
            # Ce génome possède déjà ce noeud. On en crée un neuf.
            new_id = tracker.new_node_id()

        old.enabled = False
        self.nodes[new_id] = NodeGene(new_id, NodeType.HIDDEN)
        self.add_connection(old.in_node, new_id, 1.0, tracker)
        self.add_connection(new_id, old.out_node, old.weight, tracker)
        return True

    def mutate_add_connection(self, config, tracker, rng):
        """Relie deux noeuds ensemble"""
        existing = {(c.in_node, c.out_node) for c in self.connections.values()}
        sources = list(self.nodes)
        targets = [n.id for n in self.nodes.values() if n.type in (NodeType.HIDDEN, NodeType.OUTPUT)]

        for _ in range(config.add_connection_attempts):
            a = rng.choice(sources)
            b = rng.choice(targets)
            if (a, b) in existing or self.creates_cycle(a, b):
                continue
            weight = rng.gauss(0.0, config.weight_init_stdev)
            self.add_connection(a, b, weight, tracker)
            return True
        return False

    def creates_cycle(self, a, b):
        """Ajouter a -> b crée une boucle si a est déjà atteignable depuis b."""
        if a == b:
            return True
        successors = {}
        for c in self.connections.values():
            successors.setdefault(c.in_node, []).append(c.out_node)
        to_visit, seen = [b], {b}
        while to_visit:
            node = to_visit.pop()
            if node == a:
                return True
            for nxt in successors.get(node, []):
                if nxt not in seen:
                    seen.add(nxt)
                    to_visit.append(nxt)
        return False

    def distance(self, other, config):
        """Distance entre deux génomes"""
        genes1, genes2 = self.connections, other.connections
        max1 = max(genes1,  default=-1)
        max2 = max(genes2,  default=-1)

        excess = disjoint = matching = 0
        weight_diff = 0.0
        for innov in genes1.keys() | genes2.keys():
            g1, g2 = genes1.get(innov), genes2.get(innov)
            if g1 is not None and g2 is not None:
                matching += 1
                weight_diff += abs(g1.weight - g2.weight)
            elif (g1 is not None and innov > max2) or (g2 is not None and innov > max1):
                excess += 1
            else:
                disjoint += 1

        n = max(len(genes1), len(genes2))
        if n < 20:  # pas de normalisation pour les petits génomes
            n = 1
        w = weight_diff / matching if matching else 0.0
        return config.c1 * excess / n + config.c2 * disjoint / n + config.c3 * w

    def describe(self):
        lines = ["Nœuds : " + ", ".join(f"{n.id}({n.type.value})" for n in self.nodes.values())]
        for c in self.connections.values():
            state = "" if c.enabled else "  [désactivée]"
            lines.append(f"  innov {c.innovation:2d} : {c.in_node} -> {c.out_node}  poids {c.weight:+.3f}{state}")
        return "\n".join(lines)

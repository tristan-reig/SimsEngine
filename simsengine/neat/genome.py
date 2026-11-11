import random

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
    
    def add_connection(self, in_node, out_node, weight, tracker):
        innov = tracker.connection_innovation(in_node, out_node)
        self.connections[innov] = ConnectionGene(in_node, out_node, weight, True, innov)

    def describe(self):
        lines = ["Nœuds : " + ", ".join(f"{n.id}({n.type.value})" for n in self.nodes.values())]
        for c in self.connections.values():
            state = "" if c.enabled else "  [désactivée]"
            lines.append(f"  innov {c.innovation:2d} : {c.in_node} -> {c.out_node}  poids {c.weight:+.3f}{state}")
        return "\n".join(lines)

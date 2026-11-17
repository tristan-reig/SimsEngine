import math

from .genes import NodeType


def sigmoid(x):
    z = max(-60.0, min(60.0, 4.9 * x))
    return 1.0 / (1.0 + math.exp(-z))


class FeedForwardNetwork:
    """Réseau exécutable construit à partir d'un génome."""

    def __init__(self, input_ids, bias_id, output_ids, node_evals):
        self.input_ids = input_ids
        self.bias_id = bias_id
        self.output_ids = output_ids
        self.node_evals = node_evals

    @classmethod
    def create(cls, genome):
        incoming = {}
        for c in genome.connections.values():
            if c.enabled:
                incoming.setdefault(c.out_node, []).append((c.in_node, c.weight))

        order = []
        visited = set()

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for src, _ in incoming.get(node, []):
                visit(src)
            if genome.nodes[node].type in (NodeType.HIDDEN, NodeType.OUTPUT):
                order.append(node)

        for out in genome.output_ids:
            visit(out)

        node_evals = [(n, incoming.get(n, [])) for n in order]
        return cls(genome.input_ids, genome.bias_id, genome.output_ids, node_evals)

    def activate(self, inputs):
        if len(inputs) != len(self.input_ids):
            raise ValueError(
                f"Le réseau attend {len(self.input_ids)} entrées, il en a reçu {len(inputs)}"
            )
        values = dict(zip(self.input_ids, inputs, strict=True))
        values[self.bias_id] = 1.0
        for node, links in self.node_evals:
            total = sum(values[src] * w for src, w in links)
            values[node] = sigmoid(total)
        return [values[o] for o in self.output_ids]

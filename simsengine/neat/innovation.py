class InnovationTracker:
    """Distribue les numéros d'innovation et les identifiants de nœuds."""

    def __init__(self, next_node_id) -> None:
        self.next_node_id = next_node_id
        self.next_innovation = 0
        self._connections = {}

    def connection_innovation(self, in_node, out_node):
        key = (in_node, out_node)
        if key not in self._connections:
            self._connections[key] = self.next_innovation
            self.next_innovation += 1
        return self._connections[key]

class InnovationTracker:
    """Distribue les numéros d'innovation et les identifiants de nœuds."""

    def __init__(self, next_node_id) -> None:
        self.next_node_id = next_node_id
        self.next_innovation = 0
        self._connections = {}
        self._splits = {}

    def connection_innovation(self, in_node, out_node):
        key = (in_node, out_node)
        if key not in self._connections:
            self._connections[key] = self.next_innovation
            self.next_innovation += 1
        return self._connections[key]

    def split_node_id(self, innovation):
        """Id du noeud crée en coupant la connexion `innovation`."""
        if innovation not in self._splits:
            self._splits[innovation] = self.new_node_id()
        return self._splits[innovation]

    def new_node_id(self):
        node_id = self.next_node_id
        self.next_node_id += 1
        return node_id

from dataclasses import dataclass


@dataclass
class NEATConfig:
    """Hyperparamètres de NEAT."""

    # --- Poids ---
    weight_init_stdev: float = 1.0     # écart-type d'un poids choisi au hasard
    weight_max: float = 8.0
    p_mutate_weights: float = 0.8      # chances qu'un génome mute ses poids
    p_weight_perturb: float = 0.9
    weight_perturb_stdev: float = 1.0

    # --- Mutations structurelles ---
    p_add_node: float = 0.03
    p_add_connection: float = 0.1
    add_connection_attempts: int = 20  # essais pour trouver une paire de noeuds disponible

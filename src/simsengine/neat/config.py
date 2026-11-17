from dataclasses import dataclass


@dataclass
class NEATConfig:
    """Hyperparamètres de NEAT."""

    # --- Population ---
    population_size: int = 150

    # --- Poids ---
    weight_init_stdev: float = 1.0  # écart-type d'un poids choisi au hasard
    weight_max: float = 8.0
    p_mutate_weights: float = 0.8  # chances qu'un génome mute ses poids
    p_weight_perturb: float = 0.9
    weight_perturb_stdev: float = 1.0

    # --- Mutations structurelles ---
    p_add_node: float = 0.03
    p_add_connection: float = 0.1
    add_connection_attempts: int = 20  # essais pour trouver une paire de noeuds disponible

    # --- Spéciation ---
    c1: float = 1.0
    c2: float = 1.0
    c3: float = 0.4
    compatibility_threshold: float = 3.0  # seuil de même espèce
    target_species: int = 8
    threshold_step: float = 0.1
    threshold_min: float = 0.1

    # --- Croisement ---
    p_disable_inherited: float = 0.75  # gène désactivé chez un parent -> désactivé chez l'enfant

    # --- Reproduction ---
    p_crossover: float = 0.75
    survival_threshold: float = 0.2  # part de chaque espèce autorisée à se reproduire
    elitism_min_species_size: int = 5
    stagnation_limit: int = 15
    species_elitism: int = 2  # les 2 meilleures espèces ne peuvent pas s'éteindre

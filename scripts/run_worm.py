import sys
import time

from simsengine.experiment import save_run
from simsengine.neat.config import NEATConfig
from simsengine.neat.network import FeedForwardNetwork
from simsengine.neat.population import Population
from simsengine.sim.controllers import NeatController
from simsengine.sim.simulation import simulate

N_SEGMENTS = 4
N_JOINTS = N_SEGMENTS - 1
N_INPUTS = N_JOINTS + N_SEGMENTS + 2
N_OUTPUTS = N_JOINTS


def evaluate(genomes):
    """Fitness = distance parcourue en 10 s."""
    return [simulate(NeatController(FeedForwardNetwork.create(g)), n_segments=N_SEGMENTS)
            for g in genomes]


n_generations = int(sys.argv[1]) if len(sys.argv) > 1 else 30
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

pop = Population(NEATConfig(), N_INPUTS, N_OUTPUTS, seed=seed)
start = time.perf_counter()
pop.run(evaluate, max_generations=n_generations)
champion = pop.best_ever
assert champion is not None  # au moins une génération a été évaluée
print(f"\nÉvolution terminée en {time.perf_counter() - start:.0f} s")
print(f"Meilleur ver : {champion.fitness:+.2f} m")

path = f"results/worm_seed{seed}.json"
save_run(path, champion, pop.history,
         info={"creature": "worm", "n_segments": N_SEGMENTS, "seed": seed})
print(f"\nRésultat enregistré dans {path}")
print(f"  revoir le champion  : python -m scripts.replay {path}")
print(f"  tracer les courbes  : python -m scripts.plot_run {path}")

import sys
import time

from simsengine.neat.config import NEATConfig
from simsengine.neat.network import FeedForwardNetwork
from simsengine.neat.population import Population
from simsengine.sim.controllers import NeatController
from simsengine.sim.simulation import simulate
from simsengine.viz.renderer import Renderer

N_SEGMENTS = 4
N_JOINTS = N_SEGMENTS - 1
N_INPUTS = N_JOINTS + N_SEGMENTS + 2
N_OUTPUTS = N_JOINTS


def make_controller(genome):
    return NeatController(FeedForwardNetwork.create(genome))


def evaluate(genomes):
    """Fitness = distance parcourue en 10 s."""
    return [simulate(make_controller(g), n_segments=N_SEGMENTS) for g in genomes]


n_generations = int(sys.argv[1]) if len(sys.argv) > 1 else 30

config = NEATConfig()
pop = Population(config, N_INPUTS, N_OUTPUTS, seed=0)
start = time.perf_counter()
best = pop.run(evaluate, max_generations=n_generations)
print(f"\nÉvolution terminée en {time.perf_counter() - start:.0f} s")
print(f"Meilleur ver : {best.fitness:+.2f} m")

print("\nReplay du meilleur ver (Échap pour quitter)")
renderer = Renderer(title="SimsEngine - ver évolué par NEAT")
simulate(make_controller(best), renderer=renderer, n_segments=N_SEGMENTS)
renderer.close()

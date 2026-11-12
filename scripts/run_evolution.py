import argparse
import time
from pathlib import Path

from simsengine.experiment import save_run
from simsengine.neat.config import NEATConfig
from simsengine.neat.network import FeedForwardNetwork
from simsengine.neat.population import Population
from simsengine.sim.body import load_body
from simsengine.sim.controllers import NeatController, neat_sizes
from simsengine.sim.scenarios import SCENARIOS, get_scenario
from simsengine.sim.simulation import simulate

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("body", help="fichier YAML décrivant le corps")
parser.add_argument("--scenario", choices=SCENARIOS, default="walk")
parser.add_argument("--generations", type=int, default=30)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

body = load_body(args.body)
scenario = get_scenario(args.scenario)
n_inputs, n_outputs = neat_sizes(body)
print(f"{body.name} ({scenario.name}) : {len(body.segments)} segments, "
      f"{len(body.joints)} articulations -> réseau à {n_inputs} entrées et {n_outputs} sorties\n")


def evaluate(genomes):
    """Fitness calculée par le scénario (distance parcourue en 10 s pour la marche et la nage)."""
    return [simulate(NeatController(FeedForwardNetwork.create(g)), body, scenario)
            for g in genomes]


pop = Population(NEATConfig(), n_inputs, n_outputs, seed=args.seed)
start = time.perf_counter()
pop.run(evaluate, max_generations=args.generations)
champion = pop.best_ever
assert champion is not None
print(f"\nÉvolution terminée en {time.perf_counter() - start:.0f} s")
print(f"Champion : {champion.fitness:+.2f} m")

path = Path("results") / f"{body.name}_{scenario.name}_seed{args.seed}.json"
save_run(path, champion, pop.history,
         info={"body": args.body, "scenario": scenario.name, "seed": args.seed})
print(f"\nRésultat enregistré dans {path}")
print(f"  revoir le champion  : python -m scripts.replay {path}")
print(f"  tracer les courbes  : python -m scripts.plot_run {path}")

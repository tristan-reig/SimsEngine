import argparse
import time
from pathlib import Path

from simsengine.evaluation import Evaluator
from simsengine.experiment import save_run
from simsengine.neat.config import NEATConfig
from simsengine.neat.population import Population
from simsengine.sim.body import load_body
from simsengine.sim.controllers import neat_sizes
from simsengine.sim.scenarios import SCENARIOS, get_scenario


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("body", help="fichier YAML décrivant le corps")
    parser.add_argument("--scenario", choices=SCENARIOS, default="walk")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers", type=int, default=None, help="nombre de processus (par défaut : un par coeur)"
    )
    parser.add_argument(
        "--target-species",
        type=int,
        default=NEATConfig.target_species,
        help="nombre d'espèces visé par le seuil adaptatif (0 : seuil fixe)",
    )
    args = parser.parse_args()

    body = load_body(args.body)
    scenario = get_scenario(args.scenario)
    n_inputs, n_outputs = neat_sizes(body, scenario)
    print(
        f"{body.name} ({scenario.name}) : {len(body.segments)} segments, "
        f"{len(body.joints)} articulations -> réseau à {n_inputs} entrées et {n_outputs} sorties"
    )

    config = NEATConfig(target_species=args.target_species)
    pop = Population(config, n_inputs, n_outputs, seed=args.seed)
    start = time.perf_counter()
    with Evaluator(body, scenario, workers=args.workers) as evaluate:
        print(f"Évaluation sur {evaluate.workers} processus\n")
        pop.run(evaluate, max_generations=args.generations)
    champion = pop.best_ever
    assert champion is not None
    print(f"\nÉvolution terminée en {time.perf_counter() - start:.0f} s")
    print(f"Champion : {champion.fitness:+.2f}")

    path = Path("results") / f"{body.name}_{scenario.name}_seed{args.seed}.json"
    save_run(
        path,
        champion,
        pop.history,
        info={
            "body": args.body,
            "scenario": scenario.name,
            "seed": args.seed,
            "target_species": args.target_species,
        },
    )
    print(f"\nRésultat enregistré dans {path}")
    print(f"  revoir le champion  : python -m scripts.replay {path}")
    print(f"  tracer les courbes  : python -m scripts.plot_run {path}")


if __name__ == "__main__":
    main()

import argparse
import itertools
import math
import random
import time
from pathlib import Path

from simsengine.evaluation import Evaluator
from simsengine.experiment import save_run
from simsengine.neat.config import NEATConfig
from simsengine.neat.population import Population
from simsengine.sim.body import load_body
from simsengine.sim.controllers import SineController, neat_sizes
from simsengine.sim.scenarios import SCENARIOS, get_scenario

BODIES_DIR = Path("bodies")
OUTPUT_DIR = Path("results") / "experiments"

# Les réglages comparés
CONDITIONS = {
    "adaptive": {},
    "fixed": {"target_species": 0},
    "early": {"compatibility_threshold": 1.0},
    "sine": None,
}


def run_path(body_name, scenario_name, condition, seed):
    return OUTPUT_DIR / f"{body_name}_{scenario_name}_{condition}_seed{seed}.json"


def evolve(evaluate, n_inputs, n_outputs, condition, seed, generations, label):
    """Une évolution complète. Renvoie (champion, historique)."""
    config = NEATConfig(**CONDITIONS[condition])
    pop = Population(config, n_inputs, n_outputs, seed=seed)

    def progress(pop):
        best = max(h["best"] for h in pop.history)
        generation = len(pop.history)
        print(f"\r{label} : gén. {generation}/{generations}  meilleure {best:+.2f}", end="")

    pop.run(evaluate, max_generations=generations, verbose=False, on_generation=progress)
    assert pop.best_ever is not None
    return pop.best_ever, pop.history


def random_sine(rng, n_joints):
    """Un contrôleur sinus aux paramètres tirés au hasard."""
    return SineController(
        n_joints,
        amplitude=rng.uniform(0.1, 1.0),
        frequency=rng.uniform(0.2, 3.0),
        phase=rng.uniform(-math.pi, math.pi),
    )


def sine_search(evaluate, n_joints, seed, generations, label):
    """Recherche aléatoire de sinus, avec le même budget qu'une évolution NEAT."""
    rng = random.Random(seed)
    batch_size = NEATConfig().population_size
    history = []
    best, best_fitness = None, -math.inf
    for generation in range(generations):
        controllers = [random_sine(rng, n_joints) for _ in range(batch_size)]
        scores = evaluate.evaluate_controllers(controllers)
        for controller, fitness in zip(controllers, scores, strict=True):
            if fitness > best_fitness:
                best, best_fitness = controller, fitness
        history.append(
            {"generation": generation, "best": max(scores), "mean": sum(scores) / batch_size}
        )
        print(
            f"\r{label} : lot {generation + 1}/{generations}  meilleur {best_fitness:+.2f}", end=""
        )
    assert best is not None
    params = {
        "amplitude": best.amplitude,
        "frequency": best.frequency,
        "phase": best.phase,
        "fitness": best_fitness,
    }
    return params, history


def main():
    body_names = sorted(path.stem for path in BODIES_DIR.glob("*.yaml"))
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bodies", nargs="+", choices=body_names, default=body_names)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--seeds", type=int, default=5, help="graines 0, 1, ..., N-1")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="recalcule les évolutions existantes")
    args = parser.parse_args()

    todo = {}
    skipped = 0
    for body_name, scenario_name in itertools.product(args.bodies, args.scenarios):
        for condition, seed in itertools.product(args.conditions, range(args.seeds)):
            if run_path(body_name, scenario_name, condition, seed).exists() and not args.force:
                skipped += 1
            else:
                todo.setdefault((body_name, scenario_name), []).append((condition, seed))
    total = sum(len(runs) for runs in todo.values())
    print(
        f"{total} évolutions de {args.generations} générations à lancer ({skipped} déjà faites)\n"
    )

    start = time.perf_counter()
    done = 0
    for (body_name, scenario_name), runs in todo.items():
        body = load_body(BODIES_DIR / f"{body_name}.yaml")
        scenario = get_scenario(scenario_name)
        n_inputs, n_outputs = neat_sizes(body, scenario)
        with Evaluator(body, scenario, workers=args.workers) as evaluate:
            for condition, seed in runs:
                done += 1
                label = f"[{done}/{total}] {body_name} {scenario_name} {condition} graine {seed}"
                run_start = time.perf_counter()
                info = {
                    "body": body_name,
                    "scenario": scenario_name,
                    "condition": condition,
                    "seed": seed,
                    "generations": args.generations,
                }
                if CONDITIONS[condition] is None:
                    params, history = sine_search(
                        evaluate, len(body.joints), seed, args.generations, label
                    )
                    champion, fitness = None, params["fitness"]
                    info["sine"] = params  # pour pouvoir le rejouer
                else:
                    champion, history = evolve(
                        evaluate, n_inputs, n_outputs, condition, seed, args.generations, label
                    )
                    fitness = champion.fitness
                path = run_path(body_name, scenario_name, condition, seed)
                save_run(path, champion, history, info)
                elapsed = time.perf_counter() - run_start
                print(f"\r{label} : {fitness:+.2f} en {elapsed:.0f} s" + " " * 20)

    minutes = (time.perf_counter() - start) / 60
    print(f"\nTerminé en {minutes:.1f} min. Résultats dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

import threading
from pathlib import Path

from simsengine.evaluation import Evaluator
from simsengine.experiment import save_run
from simsengine.neat.config import NEATConfig
from simsengine.neat.population import Population
from simsengine.sim.body import load_body
from simsengine.sim.controllers import neat_sizes
from simsengine.sim.scenarios import get_scenario

RESULTS_DIR = Path("results")


def next_free_seed(body_name, scenario_name):
    """Première graine sans fichier de résultat, pour ne rien écraser."""
    seed = 0
    while (RESULTS_DIR / f"{body_name}_{scenario_name}_seed{seed}.json").exists():
        seed += 1
    return seed


class LiveEvolution:
    def __init__(self, body_file, scenario_name, max_generations=50):
        self.body_file = Path(body_file)
        self.scenario_name = scenario_name
        self.max_generations = max_generations
        self.seed = next_free_seed(self.body_file.stem, scenario_name)
        self.path = RESULTS_DIR / f"{self.body_file.stem}_{scenario_name}_seed{self.seed}.json"

        self._lock = threading.Lock()
        self._history = []
        self._champion = None
        self._stop_requested = threading.Event()
        self.finished = False

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        body = load_body(self.body_file)
        scenario = get_scenario(self.scenario_name)
        n_inputs, n_outputs = neat_sizes(body, scenario)
        pop = Population(NEATConfig(), n_inputs, n_outputs, seed=self.seed)
        with Evaluator(body, scenario) as evaluate:
            pop.run(
                evaluate, self.max_generations, verbose=False, on_generation=self._on_generation
            )
        if pop.best_ever is not None:
            save_run(
                self.path,
                pop.best_ever,
                pop.history,
                info={
                    "body": str(self.body_file),
                    "scenario": self.scenario_name,
                    "seed": self.seed,
                },
            )
        self.finished = True

    def _on_generation(self, pop):
        with self._lock:
            self._history = list(pop.history)
            self._champion = pop.best_ever
        return self._stop_requested.is_set()  # Arrêt de l'évolution

    def snapshot(self):
        """(meilleur génome jusqu'ici, historique), ou (None, []) au tout début."""
        with self._lock:
            return self._champion, list(self._history)

    def stop(self):
        """Demande l'arrêt après la génération en cours, et attend qu'il ait lieu."""
        self._stop_requested.set()
        self._thread.join()

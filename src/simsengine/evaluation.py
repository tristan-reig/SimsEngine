import multiprocessing
import os

from simsengine.neat.network import FeedForwardNetwork
from simsengine.sim.controllers import NeatController
from simsengine.sim.simulation import simulate


def evaluate_controller(controller, body, scenario):
    """Fitness d'un contrôleur."""
    scores = [simulate(controller, body, v) for v in scenario.variants()]
    return sum(scores) / len(scores)


def evaluate_genome(genome, body, scenario):
    """Fitness d'un génome NEAT."""
    return evaluate_controller(NeatController(FeedForwardNetwork.create(genome)), body, scenario)


# Chaque processus de travail garde le corps et le scénario en mémoire
_worker_body = None
_worker_scenario = None


def _init_worker(body, scenario):
    global _worker_body, _worker_scenario
    _worker_body, _worker_scenario = body, scenario


def _evaluate_in_worker(genome):
    return evaluate_genome(genome, _worker_body, _worker_scenario)


def _evaluate_controller_in_worker(controller):
    return evaluate_controller(controller, _worker_body, _worker_scenario)


class Evaluator:
    """Fonction d'évaluation pour `Population.run`, en parallèle ou non."""

    def __init__(self, body, scenario, workers=None):
        self.body, self.scenario = body, scenario
        self.workers = workers if workers is not None else (os.cpu_count() or 1)
        self.pool = None
        if self.workers > 1:
            context = multiprocessing.get_context("forkserver")
            self.pool = context.Pool(
                self.workers, initializer=_init_worker, initargs=(body, scenario)
            )

    def __call__(self, genomes):
        if self.pool is None:
            return [evaluate_genome(g, self.body, self.scenario) for g in genomes]
        return self.pool.map(_evaluate_in_worker, genomes)

    def evaluate_controllers(self, controllers):
        """Comme l'appel direct, mais pour des contrôleurs déjà construits (sinus...)."""
        if self.pool is None:
            return [evaluate_controller(c, self.body, self.scenario) for c in controllers]
        return self.pool.map(_evaluate_controller_in_worker, controllers)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

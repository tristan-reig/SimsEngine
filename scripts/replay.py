"""Rejoue le champion d'une expérience enregistrée.

    python -m scripts.replay results/worm_swim_seed0.json
"""

import sys

from simsengine.experiment import load_run
from simsengine.neat.network import FeedForwardNetwork
from simsengine.sim.body import load_body
from simsengine.sim.controllers import NeatController
from simsengine.sim.scenarios import get_scenario
from simsengine.sim.simulation import simulate
from simsengine.viz.renderer import Renderer

genome, history, info = load_run(sys.argv[1])
scenario = get_scenario(info.get("scenario", "walk"))
print(f"Champion ({scenario.name}) : {genome.fitness:+.2f}, après {len(history)} générations "
      f"(graine {info['seed']})")

renderer = Renderer(title=f"SimsEngine - {sys.argv[1]}")
fitness = simulate(NeatController(FeedForwardNetwork.create(genome)),
                   load_body(info["body"]), scenario, renderer=renderer)
renderer.close()
print(f"Fitness pendant le replay : {fitness:+.2f}")

"""Rejoue le champion d'une expérience enregistrée."""

import sys

from simsengine.experiment import load_run
from simsengine.neat.network import FeedForwardNetwork
from simsengine.sim.controllers import NeatController
from simsengine.sim.simulation import simulate
from simsengine.viz.renderer import Renderer
from simsengine.sim.body import load_body

genome, history, info = load_run(sys.argv[1])
print(f"Champion : {genome.fitness:+.2f} m, après {len(history)} générations "
      f"(graine {info['seed']})")

renderer = Renderer(title=f"SimsEngine - {sys.argv[1]}")
distance = simulate(NeatController(FeedForwardNetwork.create(genome)),
                    load_body(info["body"]), renderer=renderer)
renderer.close()
print(f"Distance pendant le replay : {distance:+.2f} m")

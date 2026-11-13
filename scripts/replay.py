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
controller = NeatController(FeedForwardNetwork.create(genome))
body = load_body(info["body"])
scores = []
for i, variant in enumerate(scenario.variants(), start=1):
    scores.append(simulate(controller, body, variant, renderer=renderer))
    print(f"  épreuve {i} : {scores[-1]:+.2f}")
renderer.close()
print(f"Fitness pendant le replay (moyenne) : {sum(scores) / len(scores):+.2f}")

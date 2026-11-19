import sys

from simsengine.experiment import load_run
from simsengine.neat.network import FeedForwardNetwork
from simsengine.sim.body import load_body
from simsengine.sim.controllers import NeatController, SineController
from simsengine.sim.scenarios import get_scenario
from simsengine.sim.simulation import simulate
from simsengine.viz.renderer import Renderer

genome, history, info = load_run(sys.argv[1])
scenario = get_scenario(info.get("scenario", "walk"))
body_file = info["body"] if info["body"].endswith(".yaml") else f"bodies/{info['body']}.yaml"
body = load_body(body_file)

if genome is not None:
    controller = NeatController(FeedForwardNetwork.create(genome))
    fitness = genome.fitness
else:
    sine = info["sine"]
    controller = SineController(
        len(body.joints), sine["amplitude"], sine["frequency"], sine["phase"]
    )
    fitness = sine["fitness"]
print(
    f"Champion ({scenario.name}) : {fitness:+.2f}, après {len(history)} générations "
    f"(graine {info['seed']})"
)

renderer = Renderer(title=f"SimsEngine - {sys.argv[1]}")
scores = []
for i, variant in enumerate(scenario.variants(), start=1):  # une épreuve après l'autre
    scores.append(simulate(controller, body, variant, renderer=renderer))
    print(f"  épreuve {i} : {scores[-1]:+.2f}")
renderer.close()
print(f"Fitness pendant le replay : {scenario.combine(scores):+.2f}")

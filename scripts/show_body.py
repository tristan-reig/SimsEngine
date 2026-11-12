import sys

from simsengine.sim.body import load_body
from simsengine.sim.creature import Creature
from simsengine.sim.scenarios import WalkScenario
from simsengine.sim.simulation import physics_step
from simsengine.viz.renderer import Renderer

body = load_body(sys.argv[1])
print(f"{body.name} : {len(body.segments)} segments, {len(body.joints)} articulations")
for j in body.joints:
    print(f"  {j.a} <-> {j.b}")

scenario = WalkScenario()
space = scenario.create_world()
creature = Creature(space, body)
start_x = creature.position_x
renderer = Renderer(title=f"SimsEngine - {body.name}")
running = True
while running:
    running = renderer.handle_events()
    creature.set_targets([0.0] * len(creature.joints))  # position de départ
    physics_step(space, creature, scenario)
    renderer.center_on(creature.position_x)
    renderer.draw(space, [body.name, f"dérive : {creature.position_x - start_x:+.3f} m (doit rester ~0)"])
    renderer.tick(60)
renderer.close()

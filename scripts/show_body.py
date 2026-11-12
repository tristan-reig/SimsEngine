import sys

from simsengine.sim.body import load_body
from simsengine.sim.creature import Creature
from simsengine.sim.world import create_world
from simsengine.viz.renderer import Renderer

body = load_body(sys.argv[1])
print(f"{body.name} : {len(body.segments)} segments, {len(body.joints)} articulations")
for j in body.joints:
    print(f"  {j.a} <-> {j.b}")

space = create_world()
creature = Creature(space, body)
start_x = creature.position_x
renderer = Renderer(title=f"SimsEngine - {body.name}")
running = True
while running:
    running = renderer.handle_events()
    creature.set_targets([0.0] * len(creature.joints))
    space.step(1 / 60)
    renderer.center_on(creature.position_x)
    renderer.draw(space, [body.name, f"dérive : {creature.position_x - start_x:+.3f} m"])
    renderer.tick(60)
renderer.close()

"""Un sol, des boîtes et des balles qui tombent."""

import random

from simsengine.sim.world import add_ball, add_box, create_world
from simsengine.viz.renderer import Renderer

FPS = 60
DT = 1.0 / FPS

rng = random.Random(0)
space = create_world()

for _ in range(15):
    position = (rng.uniform(-4, 4), rng.uniform(2, 6))
    if rng.random() < 0.5:
        add_box(space, position, size=(rng.uniform(0.2, 0.6), rng.uniform(0.2, 0.6)),
                angle=rng.uniform(0, 3.14))
    else:
        add_ball(space, position, radius=rng.uniform(0.1, 0.3))

renderer = Renderer()
running = True
while running:
    running = renderer.handle_events()
    space.step(DT)
    renderer.draw(space)
    renderer.tick(FPS)
renderer.close()

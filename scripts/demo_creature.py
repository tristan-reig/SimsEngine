import math
import sys

from simsengine.sim.creature import Creature
from simsengine.sim.world import create_world
from simsengine.viz.renderer import Renderer

FPS = 60
DT = 1.0 / FPS

AMPLITUDE = 0.8
FREQUENCY = 1.0
PHASE_DEG = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0  # décalage entre articulations

space = create_world()
worm = Creature(space, n_segments=4)
start_x = worm.position_x

renderer = Renderer(title="SimsEngine")
t = 0.0
running = True
while running:
    running = renderer.handle_events()

    phase = math.radians(PHASE_DEG)
    targets = [AMPLITUDE * math.sin(2 * math.pi * FREQUENCY * t - i * phase)
               for i in range(len(worm.joints))]
    worm.set_targets(targets)

    space.step(DT)
    t += DT

    renderer.center_on(worm.position_x)  # la caméra suit le ver
    renderer.draw(space, [
        f"temps : {t:5.1f} s",
        f"distance : {worm.position_x - start_x:+.2f} m",
        f"décalage de phase : {PHASE_DEG:.0f}°",
    ])
    renderer.tick(FPS)
renderer.close()

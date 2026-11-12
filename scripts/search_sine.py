import math
import random
import sys
import time

from simsengine.sim.controllers import SineController
from simsengine.sim.simulation import simulate
from simsengine.viz.renderer import Renderer
from simsengine.sim.body import load_body

n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 300
body = load_body("bodies/worm.yaml")
rng = random.Random(0)

best_distance, best_controller = -math.inf, None
start = time.perf_counter()

for trial in range(n_trials):
    controller = SineController(
        amplitude=rng.uniform(0.1, 1.0),
        frequency=rng.uniform(0.2, 3.0),
        phase=rng.uniform(-math.pi, math.pi),
    )
    distance = simulate(controller, body)
    if distance > best_distance:
        best_distance, best_controller = distance, controller
        print(f"essai {trial:4d} : {distance:+.2f} m   (amplitude {controller.amplitude:.2f}, "
              f"fréquence {controller.frequency:.2f} Hz, phase {math.degrees(controller.phase):+.0f}°)")

elapsed = time.perf_counter() - start
print(f"\n{n_trials} simulations de 10 s en {elapsed:.1f} s "
      f"({n_trials * 10 / elapsed:.0f} fois plus vite que le temps réel)")

print("\nReplay du meilleur ver (Échap pour quitter)")
renderer = Renderer(title="SimsEngine - meilleur sinus")
simulate(best_controller, body, renderer=renderer)
renderer.close()

import math
import sys

from simsengine.sim.body import load_body
from simsengine.sim.controllers import SineController
from simsengine.sim.simulation import simulate
from simsengine.viz.renderer import Renderer

phase_deg = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
body = load_body("bodies/worm.yaml")
controller = SineController(len(body.joints), amplitude=0.75, frequency=1.0,
                            phase=math.radians(phase_deg))

renderer = Renderer(title=f"SimsEngine - ver, décalage {phase_deg:.0f}°")
distance = simulate(controller, body, renderer=renderer)
renderer.close()
print(f"Distance : {distance:+.2f} m")

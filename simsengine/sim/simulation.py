from .creature import Creature
from .scenarios import WalkScenario

DT = 1.0 / 60.0
SUBSTEPS = 4

def physics_step(space, creature, scenario):
    """Avance la physique de DT, découpé en SUBSTEPS petits pas."""
    for _ in range(SUBSTEPS):
        scenario.apply_forces(creature)
        space.step(DT / SUBSTEPS)

def simulate(controller, body, scenario=None, duration=10.0, renderer=None):
    """Simule un ver piloté par `controller` et renvoie la distance parcourue."""

    if scenario is None:
        scenario = WalkScenario()
    space = scenario.create_world()
    creature = Creature(space, body)
    start_x = creature.position_x

    n_steps = int(duration / DT)
    for step in range(n_steps):
        t = step * DT
        observation = creature.observe()
        creature.set_commands(controller.act(observation, t))
        physics_step(space, creature, scenario)

        if renderer is not None:
            if not renderer.handle_events():
                break
            follow_y = None if scenario.has_ground else creature.position_y
            renderer.center_on(creature.position_x, follow_y)
            renderer.draw(space, [f"temps : {t:5.1f} s",
                                  f"distance : {creature.position_x - start_x:+.2f} m"],
                          background=scenario.background)
            renderer.tick(60)

    return creature.position_x - start_x

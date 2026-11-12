from .creature import Creature
from .world import create_world

DT = 1.0 / 60.0


def simulate(controller, body, duration=10.0, renderer=None):
    """Simule un ver piloté par `controller` et renvoie la distance parcourue."""

    space = create_world()
    creature = Creature(space, body)
    start_x = creature.position_x

    n_steps = int(duration / DT)
    for step in range(n_steps):
        t = step * DT
        observation = creature.observe()
        creature.set_commands(controller.act(observation, t))
        space.step(DT)

        if renderer is not None:
            if not renderer.handle_events():
                break
            renderer.center_on(creature.position_x)
            renderer.draw(space, [f"temps : {t:5.1f} s",
                                  f"distance : {creature.position_x - start_x:+.2f} m"])
            renderer.tick(60)

    return creature.position_x - start_x

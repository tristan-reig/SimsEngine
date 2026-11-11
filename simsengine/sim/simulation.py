from .creature import Creature
from .world import create_world

DT = 1.0 / 60.0


def simulate(controller, duration=10.0, renderer=None, n_segments=4):
    """Simule un ver piloté par `controller` et renvoie la distance parcourue."""

    space = create_world()
    creature = Creature(space, n_segments=n_segments)
    start_x = creature.position_x
    n_joints = len(creature.joints)

    n_steps = int(duration / DT)
    for step in range(n_steps):
        t = step * DT
        observation = creature.observe()
        creature.set_targets(controller.act(observation, t, n_joints))
        space.step(DT)

        if renderer is not None:
            if not renderer.handle_events():
                break
            renderer.center_on(creature.position_x)
            renderer.draw(space, [f"temps : {t:5.1f} s",
                                  f"distance : {creature.position_x - start_x:+.2f} m"])
            renderer.tick(60)

    return creature.position_x - start_x

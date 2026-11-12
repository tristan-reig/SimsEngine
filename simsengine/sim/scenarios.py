import math

from .world import create_world


class WalkScenario:
    """Marcher le plus loin possible sur un sol plat."""

    name = "walk"
    has_ground = True
    background = (245, 245, 240)

    def create_world(self):
        return create_world()

    def apply_forces(self, creature):
        pass

    def fitness(self, creature, start_x):
        return creature.position_x - start_x


class SwimScenario:
    """Nager le plus loin possible vers la droite (pas de gravité)."""

    name = "swim"
    has_ground = False
    background = (215, 232, 245)

    def __init__(self, drag_perpendicular=20.0, drag_parallel=2.0, drag_rotation=1.0):
        self.drag_perpendicular = drag_perpendicular
        self.drag_parallel = drag_parallel
        self.drag_rotation = drag_rotation

    def create_world(self):
        return create_world(gravity=False, ground=False)

    def apply_forces(self, creature):
        kp, kl, kr = self.drag_perpendicular, self.drag_parallel, self.drag_rotation
        for body, segment in zip(creature.segments, creature.spec.segments):
            length = segment.length
            c, s = math.cos(body.angle), math.sin(body.angle)
            vx, vy = body.velocity
            v_axis = vx * c + vy * s
            v_across = -vx * s + vy * c
            fx = -length * (kl * v_axis * c - kp * v_across * s)
            fy = -length * (kl * v_axis * s + kp * v_across * c)
            body.apply_force_at_world_point((fx, fy), body.position)
            body.torque -= kr * length ** 3 * body.angular_velocity

    def fitness(self, creature, start_x):
        return creature.position_x - start_x


SCENARIOS = {"walk": WalkScenario, "swim": SwimScenario}


def get_scenario(name):
    """Crée le scénario portant ce nom."""
    if name not in SCENARIOS:
        raise ValueError(f"Scénario inconnu « {name} ». Choix possibles : {', '.join(SCENARIOS)}")
    return SCENARIOS[name]()

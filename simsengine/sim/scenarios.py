import math

import pymunk

from .world import create_world


class Scenario:
    """Comportement par défaut, que chaque scénario redéfinit"""

    name = "?"
    has_ground = True
    background = (245, 245, 245)
    n_extra_inputs = 0

    def variants(self) -> list["Scenario"]:
        """Renvoie les épreuves sur lesquelles une créature est évaluée."""
        return [self]

    def create_world(self):
        """Crée le monde physique."""
        return create_world()

    def setup(self, creature):
        """Appelé après la création de la créature"""

    def apply_forces(self, creature):
        """Forces extérieures, appliquées avant chaque petit pas de physique."""

    def extra_inputs(self, creature):
        """Capteurs supplémentaires fournis par le scénario."""
        return []

    def update(self, creature):
        """Appelé après chaque pas de simulation."""

    def fitness(self, creature, start_x):
        return creature.position_x - start_x

    def info_lines(self, creature, start_x):
        """Lignes de texte affichées pendant le replay."""
        return [f"distance : {creature.position_x - start_x:+.2f} m"]

class WalkScenario(Scenario):
    """Marcher le plus loin possible sur un sol plat."""

    name = "walk"

class SwimScenario(Scenario):
    """Nager le plus loin possible vers la droite."""

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


class FoodScenario(Scenario):
    """Manger le plus de nourriture posée sur le sol"""

    name = "food"
    n_extra_inputs = 2
    eat_radius = 0.3

    def __init__(self, first_side=+1, distance=2.0):
        self.first_side = first_side
        self.distance = distance

    def variants(self) -> list[Scenario]:
        # Une épreuve où la première boule est à droite, puis à gauche.
        return [FoodScenario(+1, self.distance), FoodScenario(-1, self.distance)]

    def create_world(self):
        space = create_world()
        self.food_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        shape = pymunk.Circle(self.food_body, 0.1)
        shape.filter = pymunk.ShapeFilter(categories=0, mask=0)
        shape.color = (230, 120, 40, 255)
        space.add(self.food_body, shape)
        self.eaten = 0
        self.side = self.first_side
        self.food_x = 0.0
        return space

    def setup(self, creature):
        self._place_food(creature)  # première boule, à côté de la créature

    def _place_food(self, creature):
        self.food_x = creature.position_x + self.side * self.distance
        self.food_body.position = (self.food_x, 0.1)

    def extra_inputs(self, creature):
        dx = self.food_x - creature.position_x
        direction = 1.0 if dx > 0 else -1.0
        closeness = min(abs(dx) / self.distance, 1.0)
        return [direction, closeness]

    def update(self, creature):
        if abs(self.food_x - creature.position_x) < self.eat_radius:
            self.eaten += 1
            self.side = -self.side
            self._place_food(creature)

    def fitness(self, creature, start_x):
        remaining = abs(self.food_x - creature.position_x)
        progress = 1.0 - remaining / self.distance
        return self.eaten + progress

    def info_lines(self, creature, start_x):
        return [f"boules mangées : {self.eaten}",
                f"nourriture à {self.food_x - creature.position_x:+.2f} m"]


SCENARIOS = {"walk": WalkScenario, "swim": SwimScenario, "food": FoodScenario}


def get_scenario(name):
    """Crée le scénario portant ce nom."""
    if name not in SCENARIOS:
        raise ValueError(f"Scénario inconnu '{name}'. Choix possibles : {', '.join(SCENARIOS)}")
    return SCENARIOS[name]()

import math

import pymunk

from .world import DT, add_ball, create_world


def _fr(value, decimals=2, sign=False):
    text = f"{value:+.{decimals}f}" if sign else f"{value:.{decimals}f}"
    return text.replace(".", ",")


class Scenario:
    """Comportement par défaut, que chaque scénario redéfinit"""

    name = "?"
    title = "?"
    unit = " m"
    has_ground = True
    background = (245, 245, 245)
    n_extra_inputs = 0

    def variants(self) -> list["Scenario"]:
        """Les épreuves sur lesquelles une créature est évaluée."""
        return [self]

    def combine(self, scores):
        """Fitness finale à partir des fitness de chaque épreuve : la moyenne."""
        return sum(scores) / len(scores)

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

    def finished(self, creature) -> bool:
        """Vrai pour arrêter l'épreuve avant la fin de sa durée."""
        return False

    def fitness(self, creature, start_x):
        return creature.position_x - start_x

    def hud_stats(self, creature, start_x):
        """Informations affichées pendant le replay."""
        return [("Distance", _fr(creature.position_x - start_x, sign=True) + " m")]


class WalkScenario(Scenario):
    """Marcher le plus loin possible sur un sol plat."""

    name = "walk"
    title = "Marche"


class SwimScenario(Scenario):
    """Nager le plus loin possible vers la droite (pas de gravité)."""

    name = "swim"
    title = "Nage"
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
        for body, segment in zip(creature.segments, creature.spec.segments, strict=True):
            length = segment.length
            c, s = math.cos(body.angle), math.sin(body.angle)
            vx, vy = body.velocity
            v_axis = vx * c + vy * s
            v_across = -vx * s + vy * c
            fx = -length * (kl * v_axis * c - kp * v_across * s)
            fy = -length * (kl * v_axis * s + kp * v_across * c)
            body.apply_force_at_world_point((fx, fy), body.position)
            body.torque -= kr * length**3 * body.angular_velocity

    def fitness(self, creature, start_x):
        return creature.position_x - start_x


class FoodScenario(Scenario):
    """Manger le plus de nourriture posée sur le sol"""

    name = "food"
    title = "Nourriture"
    unit = ""
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
        return [
            f"boules mangées : {self.eaten}",
            f"nourriture à {self.food_x - creature.position_x:+.2f} m",
        ]


class BallScenario(Scenario):
    """Pousser un ballon le plus loin possible."""

    name = "ball"
    title = "Ballon"
    n_extra_inputs = 2
    radius = 0.2
    mass = 0.4
    rolling_drag = 0.6
    approach_bonus = 0.5

    def __init__(self, side=+1, gap=1.0):
        self.side = side
        self.gap = gap

    def variants(self) -> list[Scenario]:
        return [BallScenario(+1, self.gap), BallScenario(-1, self.gap)]

    def combine(self, scores):
        # La moins bonne des deux épreuves
        return min(scores)

    def create_world(self):
        space = create_world()
        self.ball = add_ball(space, (0.0, self.radius), self.radius, self.mass)
        return space

    def setup(self, creature):
        ends = [
            body.local_to_world((sign * segment.length / 2, 0)).x
            for body, segment in zip(creature.segments, creature.spec.segments, strict=True)
            for sign in (-1, 1)
        ]
        edge = max(ends) if self.side > 0 else min(ends)
        self.ball.position = (edge + self.side * (self.gap + self.radius), self.radius)
        self.ball_start = self.ball.position.x
        self.start_distance = abs(self.ball_start - creature.position_x)
        self.closest = self.start_distance

    def apply_forces(self, creature):
        ball = self.ball
        ball.apply_force_at_world_point(-self.rolling_drag * ball.velocity, ball.position)

    def extra_inputs(self, creature):
        dx = self.ball.position.x - creature.position_x
        return [max(-1.0, min(1.0, dx / 3)), max(-1.0, min(1.0, self.ball.velocity.x / 3))]

    def update(self, creature):
        self.closest = min(self.closest, abs(self.ball.position.x - creature.position_x))

    def pushed(self):
        """Distance parcourue par le ballon dans la bonne direction (m)."""
        return self.side * (self.ball.position.x - self.ball_start)

    def fitness(self, creature, start_x):
        approach = (self.start_distance - self.closest) / self.gap  # 1 : on l'a touché
        return self.pushed() + self.approach_bonus * max(0.0, min(1.0, approach))

    def hud_stats(self, creature, start_x):
        dx = self.ball.position.x - creature.position_x
        arrow = "→" if dx > 0 else "←"
        return [
            ("Ballon poussé", _fr(self.pushed(), sign=True) + " m"),
            ("Ballon", f"{arrow} {_fr(abs(dx))} m"),
        ]


class RockScenario(Scenario):
    """Fuir un rocher qui roule derrière la créature, de plus en plus vite.

    Le rocher part `start_gap` mètres derrière la créature et accélère comme
    s'il dévalait une pente. L'épreuve s'arrête dès qu'il la touche.
    Fitness = temps de survie (s), 10 au maximum. Il faut donc démarrer vite
    et tenir l'allure : se traîner au début suffit pour être rattrapé.
    """

    name = "rock"
    title = "Rocher"
    unit = " s"
    n_extra_inputs = 2
    radius = 0.6
    start_gap = 1.5
    start_speed = 0.3
    acceleration = 0.3

    def create_world(self):
        space = create_world()
        self.rock = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        shape = pymunk.Circle(self.rock, self.radius)
        shape.friction = 1.0
        space.add(self.rock, shape)
        self.caught = False
        self.elapsed = 0.0
        return space

    def setup(self, creature):
        self.rock.position = (self._rear(creature) - self.start_gap - self.radius, self.radius)
        self.rock_start = self.rock.position.x
        self.apply_forces(creature)  # le rocher roule dès le départ

    def _rear(self, creature):
        return min(
            body.local_to_world((sign * segment.length / 2, 0)).x
            for body, segment in zip(creature.segments, creature.spec.segments, strict=True)
            for sign in (-1, 1)
        )

    def lead(self, creature):
        return self._rear(creature) - (self.rock.position.x + self.radius)

    def speed(self):
        travelled = self.rock.position.x - self.rock_start
        return math.sqrt(self.start_speed**2 + 2 * self.acceleration * travelled)

    def apply_forces(self, creature):
        speed = self.speed()
        self.rock.velocity = (speed, 0.0)
        self.rock.angular_velocity = -speed / self.radius  # il roule sans glisser

    def extra_inputs(self, creature):
        return [max(0.0, min(1.0, self.lead(creature) / 3)), min(1.0, self.speed() / 3)]

    def update(self, creature):
        if self.caught:
            return
        self.elapsed += DT

        def touch(arbiter):
            self.caught = True

        self.rock.each_arbiter(touch)

    def finished(self, creature):
        return self.caught

    def fitness(self, creature, start_x):
        return self.elapsed

    def hud_stats(self, creature, start_x):
        return [
            ("Avance", _fr(max(0.0, self.lead(creature))) + " m"),
            ("Rocher", _fr(self.speed(), 1) + " m/s"),
        ]


SCENARIOS = {
    "walk": WalkScenario,
    "swim": SwimScenario,
    "food": FoodScenario,
    "ball": BallScenario,
    "rock": RockScenario,
}


def get_scenario(name):
    """Crée le scénario portant ce nom."""
    if name not in SCENARIOS:
        raise ValueError(f"Scénario inconnu '{name}'. Choix possibles : {', '.join(SCENARIOS)}")
    return SCENARIOS[name]()

import pymunk

DT = 1.0 / 60.0
SUBSTEPS = 4
GRAVITY = -9.81
GROUND_THICKNESS = 0.1


def create_world(gravity=True, ground=True):
    """Crée un espace physique avec la gravité et un sol plat en y = 0."""
    space = pymunk.Space()
    space.gravity = (0.0, GRAVITY if gravity else 0.0)
    space.collision_slop = 0.002
    if not ground:
        return space

    r = GROUND_THICKNESS / 2
    ground = pymunk.Segment(space.static_body, (-1000.0, -r), (1000.0, -r), r)
    ground.friction = 1.0
    ground.elasticity = 1.0
    space.add(ground)
    return space


def add_box(space, position, size, mass=1.0, angle=0.0):
    """Ajoute une boîte dynamique"""
    moment = pymunk.moment_for_box(mass, size)
    body = pymunk.Body(mass, moment)
    body.position = position
    body.angle = angle
    shape = pymunk.Poly.create_box(body, size)
    shape.friction = 0.8
    space.add(body, shape)
    return body


def add_ball(space, position, radius, mass=1.0):
    """Ajoute une balle dynamique."""
    moment = pymunk.moment_for_circle(mass, 0.0, radius)
    body = pymunk.Body(mass, moment)
    body.position = position
    shape = pymunk.Circle(body, radius)
    shape.friction = 0.8
    shape.elasticity = 0.6
    space.add(body, shape)
    return body

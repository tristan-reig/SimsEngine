import pymunk
from pymunk.constraints import PivotJoint, RotaryLimitJoint, SimpleMotor


def touches_ground(body):
    """Vrai si le corps est en contact avec le sol"""
    touching = False

    def check(arbiter):
        nonlocal touching
        if any(s.body.body_type == pymunk.Body.STATIC for s in arbiter.shapes):
            touching = True

    body.each_arbiter(check)
    return touching


class Joint:
    """Articulation motorisée entre deux segments voisins."""

    def __init__(self, space, body_a, body_b, spec):
        self.a, self.b = body_a, body_b
        self.min_angle, self.max_angle = spec.min_angle, spec.max_angle
        self.gain = spec.gain
        self.max_speed = spec.max_speed
        self.rest_angle = body_b.angle - body_a.angle
        
        pivot = PivotJoint(body_a, body_b, spec.anchor)
        limit = RotaryLimitJoint(body_a, body_b,
                                 self.rest_angle + spec.min_angle,
                                 self.rest_angle + spec.max_angle)
        self.motor = SimpleMotor(body_a, body_b, 0.0)
        self.motor.max_force = spec.max_torque
        space.add(pivot, limit, self.motor)

    @property
    def angle(self):
        """Angle de l'articulation"""
        return self.b.angle - self.a.angle - self.rest_angle

    @property
    def normalized_angle(self):
        """Angle ramené entre -1 et 1."""
        middle = (self.min_angle + self.max_angle) / 2
        half_range = (self.max_angle - self.min_angle) / 2
        return (self.angle - middle) / half_range

    def set_command(self, command):
        command = max(-1.0, min(1.0, command))
        middle = (self.min_angle + self.max_angle) / 2
        half_range = (self.max_angle - self.min_angle) / 2
        self.set_target(middle + command * half_range)

    def set_target(self, target):
        """Fait tendre l'articulation vers un angle"""
        target = max(self.min_angle, min(self.max_angle, target))
        speed = self.gain * (target - self.angle)
        speed = max(-self.max_speed, min(self.max_speed, speed))
        self.motor.rate = -speed

class Creature:
    """Construction d'une créature à partir d'une description de son corps"""
    def __init__(self, space, spec) -> None:
        self.spec = spec
        self.segments = []
        self.joints = []
        bodies_by_name = {}
        no_self_collision = pymunk.ShapeFilter(group=1)

        for seg in spec.segments:
            size = (seg.length, seg.width)
            body = pymunk.Body(seg.mass, pymunk.moment_for_box(seg.mass, size))
            body.position = seg.center
            body.angle = seg.angle
            shape = pymunk.Poly.create_box(body, size)
            shape.friction = seg.friction
            shape.filter = no_self_collision
            space.add(body, shape)
            self.segments.append(body)
            bodies_by_name[seg.name] = body

        for j in spec.joints:
            self.joints.append(Joint(space, bodies_by_name[j.a], bodies_by_name[j.b], j))

    @property
    def position_x(self):
        """Abscisse moyenne des segments pour mesurer la distance parcourue."""
        return sum(b.position.x for b in self.segments) / len(self.segments)

    @property
    def position_y(self):
        """Ordonnée moyenne des segments."""
        return sum(b.position.y for b in self.segments) / len(self.segments)

    def observe(self):
        """Ce que perçoit la créature"""
        angles = [j.normalized_angle for j in self.joints]
        contacts = [1.0 if touches_ground(b) else 0.0 for b in self.segments]
        return angles + contacts

    def set_commands(self, commands):
        for joint, command in zip(self.joints, commands):
            joint.set_command(command)

    def set_targets(self, targets):
        for joint, target in zip(self.joints, targets):
            joint.set_target(target)

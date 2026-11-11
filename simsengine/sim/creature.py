import math
import pymunk

from pymunk.constraints import PivotJoint, RotaryLimitJoint, SimpleMotor


class Joint:
    """Articulation motorisée entre deux segments voisins."""

    def __init__(self, space, body_a, body_b, anchor, max_angle, max_torque, gain) -> None:
        self.a, self.b = body_a, body_b
        self.gain = gain
        pivot = PivotJoint(body_a, body_b, anchor)
        limit = RotaryLimitJoint(body_a, body_b, -max_angle, max_angle)
        self.motor = SimpleMotor(body_a, body_b, 0.0)
        self.motor.max_force = max_torque
        space.add(pivot , limit, self.motor)
        self.max_angle = max_angle

    @property
    def angle(self):
        return self.b.angle - self.a.angle

    def set_target(self, target):
        """Fait tendre l'articulation vers un angle"""
        target = max(-self.max_angle, min(self.max_angle, target))
        self.motor.rate = -self.gain * (target - self.angle)

class Creature:
    def __init__(self, space, x=0.0, n_segments=3, segment_length=0.5, segment_width=0.1,
                 segment_mass=1.0, max_angle=math.radians(60), max_torque=20.0, gain=25.0) -> None:
        self.segments = []
        self.joints = []
        # Les segments d'une même créature ne se cognent pas entre eux
        no_self_collision = pymunk.ShapeFilter(group=1)
        y = segment_width / 2 + 0.01  # positionnement au-dessus du sol

        for i in range(n_segments):
            size = (segment_length, segment_width)
            body = pymunk.Body(segment_mass, pymunk.moment_for_box(segment_mass, size))
            body.position = (x + i * segment_length, y)
            shape = pymunk.Poly.create_box(body, size)
            shape.friction = 1.0
            shape.filter = no_self_collision
            space.add(body, shape)
            self.segments.append(body)

            if i > 0:
                anchor = (x + (i - 0.5) * segment_length, y)
                self.joints.append(Joint(space, self.segments[i - 1], body, anchor,
                                         max_angle, max_torque, gain))

    @property
    def position_x(self):
        """Abscisse moyenne des segments pour mesurer la distance parcourue."""
        return sum(b.position.x for b in self.segments) / len(self.segments)

    def set_targets(self, targets):
        for joint, target in zip(self.joints, targets):
            joint.set_target(target)

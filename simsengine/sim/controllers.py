import math


class SineController:
    """Contrôleur sans cerveau"""

    def __init__(self, amplitude, frequency, phase) -> None:
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def act(self, observation, t, n_joints):
        """Renvoie l'angle cible de chaque articulation au temps t."""
        return [self.amplitude * math.sin(2 * math.pi * self.frequency * t - i * self.phase)
                for i in range(n_joints)]

class NeatController:
    """Contrôleur piloté par un réseau de neurones géré par NEAT."""

    def __init__(self, network, clock_frequency=2.0, max_angle=math.radians(60)) -> None:
        self.network = network
        self.clock_frequency = clock_frequency
        self.max_angle = max_angle

    def act(self, observation, t, n_joints):
        phase = 2 * math.pi * self.clock_frequency * t
        inputs = observation + [math.sin(phase), math.cos(phase)]
        outputs = self.network.activate(inputs)
        return [(2 * o - 1) * self.max_angle for o in outputs]

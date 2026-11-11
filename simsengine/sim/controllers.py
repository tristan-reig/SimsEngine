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

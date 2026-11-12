"""Contrôleurs : ils décident à chaque pas des commandes des articulations.

Un contrôleur a une méthode `act(observation, t)` qui renvoie une commande
par articulation, entre -1 et +1 : -1 = butée min, 0 = milieu, +1 = butée max.
C'est la créature qui convertit ces commandes en angles, selon les butées
de chaque articulation. Un même contrôleur marche donc pour n'importe quel corps.
"""

import math


def neat_sizes(body):
    """Nombre d'entrées et de sorties du réseau NEAT pour un corps donné."""
    n_inputs = len(body.joints) + len(body.segments) + 2  # angles + contacts + horloge
    n_outputs = len(body.joints)                           # une commande par articulation
    return n_inputs, n_outputs


class SineController:
    """Contrôleur sans cerveau : chaque articulation suit un sinus.

    Il ignore les capteurs. Il sert de point de comparaison avant NEAT.
    """

    def __init__(self, n_joints, amplitude, frequency, phase):
        self.n_joints = n_joints
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def act(self, observation, t):
        return [self.amplitude * math.sin(2 * math.pi * self.frequency * t - i * self.phase)
                for i in range(self.n_joints)]


class NeatController:
    """Contrôleur piloté par un réseau de neurones dirigé par NEAT."""

    def __init__(self, network, clock_frequency=2.0):
        self.network = network
        self.clock_frequency = clock_frequency  # Hz

    def act(self, observation, t):
        phase = 2 * math.pi * self.clock_frequency * t
        inputs = observation + [math.sin(phase), math.cos(phase)]
        return [2 * o - 1 for o in self.network.activate(inputs)]

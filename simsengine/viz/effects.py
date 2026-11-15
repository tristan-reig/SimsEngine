import math
import random
from collections import deque

from .drawing import circle, ring, with_alpha


class Trail:
    """Les dernières positions de la créature, dessinées en points qui s'estompent."""

    def __init__(self, length=30, every=3):
        self.points = deque(maxlen=length)
        self.every = every
        self.frame = 0

    def reset(self, length=None):
        if length is not None:
            self.points = deque(maxlen=length)
        else:
            self.points.clear()
        self.frame = 0

    def add(self, x, y):
        self.frame += 1
        if self.frame % self.every == 0:
            self.points.append((x, y))

    def draw(self, surface, to_screen, color):
        n = len(self.points)
        for i, (x, y) in enumerate(self.points):
            alpha = 40 + 120 * (i + 1) / n
            circle(surface, with_alpha(color, alpha), to_screen(x, y), 3)


class Particles:
    """Petit système de particules : bulles, gerbes, textes flottants."""

    def __init__(self, seed=0):
        self.items = []
        self.rng = random.Random(seed)

    def clear(self):
        self.items.clear()

    def bubble(self, x, y):
        """Une bulle qui remonte en oscillant."""
        self.items.append({"kind": "bubble", "x": x, "y": y,
                           "vx": self.rng.uniform(-0.05, 0.05), "vy": self.rng.uniform(0.25, 0.45),
                           "life": 0.0, "duration": self.rng.uniform(1.8, 2.8),
                           "radius": self.rng.uniform(2.0, 5.0), "phase": self.rng.uniform(0, 6.28)})

    def burst(self, x, y, color, count=14):
        """Une gerbe de petites particules dans toutes les directions."""
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(0.4, 1.2)
            self.items.append({"kind": "spark", "x": x, "y": y, "color": color,
                               "vx": speed * math.cos(angle), "vy": speed * math.sin(angle) + 0.5,
                               "life": 0.0, "duration": self.rng.uniform(0.5, 0.9),
                               "radius": self.rng.uniform(2.0, 3.5)})

    def text(self, x, y, text, color):
        """Un texte qui monte en s'effaçant."""
        self.items.append({"kind": "text", "x": x, "y": y, "text": text, "color": color,
                           "vx": 0.0, "vy": 0.35, "life": 0.0, "duration": 1.2})

    def update(self, dt, gravity=0.0):
        for p in self.items:
            p["life"] += dt
            if p["kind"] == "spark":
                p["vy"] -= gravity * dt
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
        self.items = [p for p in self.items if p["life"] < p["duration"]]

    def draw(self, surface, to_screen, font):
        for p in self.items:
            fade = 1.0 - p["life"] / p["duration"]
            sx, sy = to_screen(p["x"], p["y"])
            if p["kind"] == "bubble":
                sx += 3 * math.sin(p["phase"] + 3 * p["life"])
                ring(surface, (255, 255, 255, round(170 * fade)), (sx, sy), p["radius"], 1)
            elif p["kind"] == "spark":
                circle(surface, with_alpha(p["color"], 220 * fade), (sx, sy), p["radius"])
            else:
                image = font.render(p["text"], True, p["color"])
                image.set_alpha(round(255 * fade))
                surface.blit(image, (sx - image.get_width() / 2, sy))

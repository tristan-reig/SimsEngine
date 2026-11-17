import math
import random

import pygame

from .drawing import circle, ellipse, polygon, vertical_gradient

SKY = [(0.0, (207, 229, 243)), (0.75, (233, 242, 241)), (1.0, (243, 246, 236))]
HILLS_FAR = (201, 222, 192)
HILLS_NEAR = (176, 208, 164)
GRASS = (125, 179, 106)
SOIL = (200, 164, 126)
SOIL_EDGE = (176, 140, 102)
PEBBLE = (183, 150, 112)
RULER = (111, 86, 64)
RULER_TEXT = (94, 72, 54)
WATER = [
    (0.0, (169, 218, 238)),
    (0.45, (111, 178, 214)),
    (0.8, (62, 129, 174)),
    (1.0, (45, 102, 144)),
]

CLOUD_TILE = 1400
CLOUDS = [(170, 120, 1.0), (720, 80, 1.1), (1150, 150, 0.8)]


def _hill_y(u, base, amplitude, seed):
    """Hauteur à l'écran d'une ligne de collines, en fonction de l'abscisse u."""
    return (
        base
        + amplitude * math.sin(u * 0.0045 + seed)
        + amplitude * 0.5 * math.sin(u * 0.011 + 2 * seed)
    )


class LandScenery:
    def __init__(self, width, ground_y):
        self.width, self.ground_y = width, ground_y
        self.sky = vertical_gradient((width, ground_y), SKY)

    def draw_background(self, surface, camera_px):
        """Ciel, nuages et collines."""
        surface.blit(self.sky, (0, 0))

        offset = (camera_px * 0.1) % CLOUD_TILE  # les nuages bougent à peine
        for x, y, size in CLOUDS:
            for tile in (0, CLOUD_TILE):
                cx = x - offset + tile
                if -120 < cx < self.width + 120:
                    self._cloud(surface, cx, y, size)

        for factor, base, amplitude, color, seed in (
            (0.3, self.ground_y - 100, 22, HILLS_FAR, 0.0),
            (0.6, self.ground_y - 52, 14, HILLS_NEAR, 1.7),
        ):
            shift = camera_px * factor
            points = [
                (sx, _hill_y(sx + shift, base, amplitude, seed))
                for sx in range(0, self.width + 20, 20)
            ]
            points += [(self.width, self.ground_y), (0, self.ground_y)]
            polygon(surface, color, points)

    @staticmethod
    def _cloud(surface, x, y, size):
        color = (255, 255, 255, 230)
        ellipse(surface, color, (x, y), 58 * size, 22 * size)
        ellipse(surface, color, (x + 40 * size, y - 12 * size), 40 * size, 24 * size)
        ellipse(surface, color, (x - 35 * size, y + 8 * size), 34 * size, 16 * size)

    def draw_ground(self, surface, camera_x, ppm, font):
        """Herbe, terre, cailloux et règle graduée en mètres."""
        gy = self.ground_y
        pygame.draw.rect(surface, GRASS, (0, gy, self.width, 14))
        pygame.draw.rect(surface, SOIL, (0, gy + 14, self.width, surface.get_height() - gy - 14))
        pygame.draw.rect(surface, SOIL_EDGE, (0, gy + 14, self.width, 4))

        half_width_m = self.width / 2 / ppm
        for m in range(
            math.floor(camera_x - half_width_m) - 1, math.ceil(camera_x + half_width_m) + 2
        ):
            sx = self.width / 2 + (m - camera_x) * ppm
            rng = random.Random(m)
            for _ in range(2):
                circle(
                    surface,
                    PEBBLE,
                    (sx + rng.uniform(0, ppm), gy + rng.uniform(30, 105)),
                    rng.uniform(2, 3.2),
                )
            pygame.draw.line(surface, RULER, (sx, gy), (sx, gy + 28), 2)
            label = font.render(f"{m} m".replace("-", "−"), True, RULER_TEXT)
            surface.blit(label, (sx + 6, gy + 32))


class WaterScenery:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.water = vertical_gradient((width, height), WATER)
        self.rays = pygame.Surface((width, height), pygame.SRCALPHA)

    def draw_background(self, surface, camera_x, camera_y, ppm, t):
        surface.blit(self.water, (0, 0))

        # Rayons de lumière qui ondulent doucement
        self.rays.fill((0, 0, 0, 0))
        sway = 25 * math.sin(0.3 * t)
        for x, w in ((180, 80), (420, 50), (700, 80)):
            x += sway
            polygon(
                self.rays,
                (255, 255, 255, 26),
                [(x, 0), (x + w, 0), (x + w - 140, self.height), (x - 160, self.height)],
            )
        surface.blit(self.rays, (0, 0))

        # Particules en suspension, fixes dans l'eau.
        half_w, half_h = self.width / 2 / ppm, self.height / 2 / ppm
        for ix in range(math.floor(camera_x - half_w) - 1, math.ceil(camera_x + half_w) + 1):
            for iy in range(math.floor(camera_y - half_h) - 1, math.ceil(camera_y + half_h) + 1):
                rng = random.Random(ix * 7919 + iy * 104729)
                for _ in range(2):
                    wx, wy = ix + rng.random(), iy + rng.random()
                    sx = self.width / 2 + (wx - camera_x) * ppm
                    sy = self.height / 2 - (wy - camera_y) * ppm
                    circle(surface, (255, 255, 255, 90), (sx, sy), rng.uniform(1.2, 1.9))

    def draw_ruler(self, surface, camera_x, ppm, font):
        """Règle graduée en bas de l'écran (pas de sol dans l'eau), au-dessus des touches."""
        y = self.height - 76
        half_width_m = self.width / 2 / ppm
        for m in range(
            math.floor(camera_x - half_width_m) - 1, math.ceil(camera_x + half_width_m) + 2
        ):
            sx = self.width / 2 + (m - camera_x) * ppm
            pygame.draw.line(surface, (220, 236, 247), (sx, y), (sx, y + 16), 2)
            label = font.render(f"{m} m".replace("-", "−"), True, (230, 242, 250))
            surface.blit(label, (sx + 6, y + 2))

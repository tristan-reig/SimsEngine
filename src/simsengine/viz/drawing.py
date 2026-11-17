import math
from itertools import pairwise

import pygame
import pygame.gfxdraw


def lighten(color, amount):
    """Éclaircit la couleur."""
    return tuple(round(c + (255 - c) * amount) for c in color[:3])


def darken(color, factor):
    """Assombrit la couleur."""
    return tuple(round(c * factor) for c in color[:3])


def with_alpha(color, alpha):
    return (*color[:3], max(0, min(255, round(alpha))))


def vertical_gradient(size, stops):
    """Surface remplie d'un dégradé vertical."""
    width, height = size
    surface = pygame.Surface(size)
    for y in range(height):
        t = y / max(1, height - 1)
        for (t0, c0), (t1, c1) in pairwise(stops):
            if t0 <= t <= t1:
                k = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                color = tuple(round(a + (b - a) * k) for a, b in zip(c0, c1, strict=True))
                break
        else:
            color = stops[-1][1]
        pygame.draw.line(surface, color, (0, y), (width, y))
    return surface


def circle(surface, color, center, radius):
    """Disque plein à bord lissé."""
    x, y, r = round(center[0]), round(center[1]), round(radius)
    if r < 1:
        return
    pygame.gfxdraw.filled_circle(surface, x, y, r, color)
    pygame.gfxdraw.aacircle(surface, x, y, r, color)


def ring(surface, color, center, radius, thickness=2):
    """Cercle vide."""
    x, y = round(center[0]), round(center[1])
    for i in range(thickness):
        pygame.gfxdraw.aacircle(surface, x, y, round(radius) - i, color)


def polygon(surface, color, points):
    """Polygone plein à bord lissé."""
    pts = [(round(x), round(y)) for x, y in points]
    pygame.gfxdraw.filled_polygon(surface, pts, color)
    pygame.gfxdraw.aapolygon(surface, pts, color)


def ellipse(surface, color, center, rx, ry):
    x, y = round(center[0]), round(center[1])
    if rx < 1 or ry < 1:
        return
    pygame.gfxdraw.filled_ellipse(surface, x, y, round(rx), round(ry), color)
    pygame.gfxdraw.aaellipse(surface, x, y, round(rx), round(ry), color)


def capsule(surface, color, p1, p2, width):
    """Segment épais aux bouts arrondis, de p1 à p2."""
    (x1, y1), (x2, y2) = p1, p2
    half = width / 2
    length = math.hypot(x2 - x1, y2 - y1)
    if length > 1e-6:
        nx, ny = -(y2 - y1) / length * half, (x2 - x1) / length * half
        polygon(
            surface,
            color,
            [(x1 + nx, y1 + ny), (x2 + nx, y2 + ny), (x2 - nx, y2 - ny), (x1 - nx, y1 - ny)],
        )
    circle(surface, color, p1, half)
    circle(surface, color, p2, half)


def rounded_panel(surface, rect, color, radius=12):
    """Rectangle arrondi."""
    rect = pygame.Rect(rect)
    if len(color) == 4 and color[3] < 255:
        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(panel, color, panel.get_rect(), border_radius=radius)
        surface.blit(panel, rect.topleft)
    else:
        pygame.draw.rect(surface, color[:3], rect, border_radius=radius)


def format_number(value, decimals=2, sign=False):
    text = f"{value:+.{decimals}f}" if sign else f"{value:.{decimals}f}"
    return text.replace(".", ",")

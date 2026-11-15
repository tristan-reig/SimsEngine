import math

import pygame
import pymunk
import pymunk.pygame_util

from .drawing import (capsule, circle, darken, ellipse, format_number, lighten,
                      ring, rounded_panel, with_alpha)
from .effects import Particles, Trail
from .scenery import LandScenery, WaterScenery

FONT_NAMES = "inter,dejavusans,liberationsans,arial"
TEXT = (31, 42, 36)
MUTED = (91, 102, 95)
PANEL = (255, 255, 255, 222)
JOINT = (255, 246, 232)
FOOD = (242, 160, 61)
FOOD_EDGE = (201, 122, 28)
SHADOW = (47, 59, 42)
DEBUG_BACKGROUND = (245, 245, 240)

CAMERA_SMOOTHING = 0.12  # part de l'écart rattrapée par la caméra à chaque image


class Renderer:
    def __init__(self, width=1000, height=600, pixels_per_meter=100, title="SimsEngine"):
        # Initialisation de l'affichage et du texte
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.width, self.height = width, height
        self.ppm = pixels_per_meter
        self.ground_y = height - 120

        self.fonts = {
            "small": pygame.font.SysFont(FONT_NAMES, 13),
            "small_bold": pygame.font.SysFont(FONT_NAMES, 13, bold=True),
            "body": pygame.font.SysFont(FONT_NAMES, 15),
            "bold": pygame.font.SysFont(FONT_NAMES, 15, bold=True),
            "value": pygame.font.SysFont(FONT_NAMES, 21, bold=True),
            "title": pygame.font.SysFont(FONT_NAMES, 38, bold=True),
        }
        self.font = self.fonts["body"]

        self.camera_x, self.camera_y = 0.0, 0.0
        self.anchor_y = self.ground_y
        self.debug = False

        self.land = LandScenery(width, self.ground_y)
        self.water = WaterScenery(width, height)
        self.trail = Trail()
        self.particles = Particles()
        self._sim = None
        self._lastStep = -1
        self._eaten = 0
        self._last_food = None

        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        # Dessin des formes et des articulations
        self.draw_options.flags = (
            pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
            | pymunk.SpaceDebugDrawOptions.DRAW_CONSTRAINTS
        )
        self.center_on(0.0)

    # ------------------------------------------------------------------
    # Caméra
    # ------------------------------------------------------------------
    def to_screen(self, x, y):
        """Coordonnées du monde -> coordonnées de l'écran."""
        return (self.width / 2 + (x - self.camera_x) * self.ppm,
                self.anchor_y - (y - self.camera_y) * self.ppm)

    def center_on(self, x, y=None):
        """Place la caméra sur l'abscisse x."""
        self.camera_x = x
        if y is None:
            self.camera_y, self.anchor_y = 0.0, self.ground_y
        else:
            self.camera_y, self.anchor_y = y, self.height / 2

    def _follow(self, x, y=None):
        self.camera_x += (x - self.camera_x) * CAMERA_SMOOTHING
        if y is None:
            self.camera_y, self.anchor_y = 0.0, self.ground_y
        else:
            self.camera_y += (y - self.camera_y) * CAMERA_SMOOTHING
            self.anchor_y = self.height / 2

    # ------------------------------------------------------------------
    # Rendu de la simulation
    # ------------------------------------------------------------------
    def draw_world(self, sim):
        """Dessine le décor, la créature et les effets."""
        scenario, creature = sim.scenario, sim.creature
        if sim is not self._sim:  # On repart de zéro
            self._start(sim)

        follow_y = None if scenario.has_ground else creature.position_y
        self._follow(creature.position_x, follow_y)
        if sim.step_count != self._last_step:  # les effets avancent avec la simulation
            self._last_step = sim.step_count
            self._update_effects(sim)

        if self.debug:
            self.screen.fill(DEBUG_BACKGROUND)
            self._debug_space(sim.space)
            return

        color = creature.spec.color
        if scenario.has_ground:
            self.land.draw_background(self.screen, self.camera_x * self.ppm)
            self.land.draw_ground(self.screen, self.camera_x, self.ppm, self.fonts["small"])
        else:
            self.water.draw_background(self.screen, self.camera_x, self.camera_y, self.ppm,
                                       pygame.time.get_ticks() / 1000)
        if hasattr(scenario, "food_x"):
            self._draw_food(scenario.food_x)
        if scenario.has_ground:
            self._draw_shadow(creature)
        self.trail.draw(self.screen, self.to_screen, color)
        self.draw_creature(creature, color)
        self.particles.draw(self.screen, self.to_screen, self.fonts["bold"])
        if not scenario.has_ground:
            self.water.draw_ruler(self.screen, self.camera_x, self.ppm, self.fonts["small"])

    def _start(self, sim):
        self._sim = sim
        self._last_step = -1
        self.center_on(sim.creature.position_x,
                       None if sim.scenario.has_ground else sim.creature.position_y)
        has_food = hasattr(sim.scenario, "food_x")
        self.trail.reset(length=120 if has_food else 30)  # Visualisation les allers-retours
        self.trail.every = 4 if has_food else 3
        self.particles.clear()
        self._eaten = getattr(sim.scenario, "eaten", 0)
        self._last_food = None

    def _update_effects(self, sim):
        scenario, creature = sim.scenario, sim.creature
        trail_y = 0.03 if scenario.has_ground else creature.position_y
        self.trail.add(creature.position_x, trail_y)
        self.particles.update(1 / 60, gravity=2.0 if scenario.has_ground else 0.0)
        if not scenario.has_ground and sim.step_count % 5 == 0:
            tail = creature.segments[0].local_to_world((-creature.spec.segments[0].length / 2, 0))
            self.particles.bubble(tail.x, tail.y)
        if hasattr(scenario, "food_x"):
            if scenario.eaten > self._eaten and self._last_food is not None:
                x, y = self._last_food  # Endroit de la boule mangée
                self.particles.burst(x, y, FOOD)
                self.particles.text(x, y + 0.3, "+1", darken(FOOD, 0.6))
            self._eaten = scenario.eaten
            self._last_food = (scenario.food_x, 0.1)

    def draw_creature(self, creature, color):
        """Segments en capsules arrondies avec contour et reflet."""
        outline, shine = darken(color, 0.68), lighten(color, 0.3)
        for body, segment in sorted(zip(creature.segments, creature.spec.segments),
                                    key=lambda pair: pair[1].width):
            half = segment.length / 2
            p1 = self.to_screen(*body.local_to_world((-half, 0)))
            p2 = self.to_screen(*body.local_to_world((half, 0)))
            width = segment.width * self.ppm
            capsule(self.screen, outline, p1, p2, width + 5)
            capsule(self.screen, color, p1, p2, width)
            if width >= 10:
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                length = math.hypot(dx, dy) or 1.0
                nx, ny = -dy / length, dx / length
                if ny > 0:
                    nx, ny = -nx, -ny
                off = width * 0.22
                q1 = (p1[0] + dx * 0.1 + nx * off, p1[1] + dy * 0.1 + ny * off)
                q2 = (p2[0] - dx * 0.1 + nx * off, p2[1] - dy * 0.1 + ny * off)
                capsule(self.screen, shine, q1, q2, width * 0.28)
        radius = max(4, 0.04 * self.ppm)
        for joint in creature.joints:
            center = self.to_screen(*joint.position)
            circle(self.screen, JOINT, center, radius)
            ring(self.screen, outline, center, radius, 2)

    def _draw_shadow(self, creature):
        xs = [self.to_screen(*b.position)[0] for b in creature.segments]
        height = max(0.0, min(b.position.y for b in creature.segments))
        fade = max(0.0, 1.0 - height / 1.5)
        rx = (max(xs) - min(xs)) / 2 + 0.25 * self.ppm
        ellipse(self.screen, with_alpha(SHADOW, 45 * fade),
                ((max(xs) + min(xs)) / 2, self.ground_y + 3), rx * (0.7 + 0.3 * fade), 6)

    def _draw_food(self, food_x):
        cx, cy = self.to_screen(food_x, 0.1)
        pulse = math.sin(pygame.time.get_ticks() / 1000 * 3)
        ellipse(self.screen, with_alpha(SHADOW, 45), (cx, self.ground_y + 3), 18, 4)
        ring(self.screen, with_alpha(FOOD, 90 + 40 * pulse), (cx, cy), 0.17 * self.ppm + 3 * pulse, 2)
        radius = 0.1 * self.ppm
        circle(self.screen, FOOD_EDGE, (cx, cy), radius + 2)
        circle(self.screen, FOOD, (cx, cy), radius)
        circle(self.screen, (255, 255, 255, 140), (cx - radius * 0.35, cy - radius * 0.35), radius * 0.32)

    def _debug_space(self, space):
        """Dessin brut de pymunk avec la même caméra, plus des repères tous les mètres."""
        self.draw_options.transform = pymunk.Transform(
            a=self.ppm, b=0, c=0, d=-self.ppm,
            tx=self.width / 2 - self.camera_x * self.ppm,
            ty=self.anchor_y + self.camera_y * self.ppm)
        space.debug_draw(self.draw_options)
        ground_y = self.to_screen(0, 0)[1]
        half_width_m = self.width / 2 / self.ppm
        for m in range(math.floor(self.camera_x - half_width_m) - 1,
                       math.ceil(self.camera_x + half_width_m) + 2):
            sx = self.width / 2 + (m - self.camera_x) * self.ppm
            pygame.draw.line(self.screen, (90, 90, 90), (sx, ground_y), (sx, ground_y + 22), 2)
            label = self.fonts["small"].render(f"{m} m", True, (120, 120, 120))
            self.screen.blit(label, (sx + 4, ground_y + 24))

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------
    def draw_card(self, x, y, width, title, stats, footer=None, dot=None, badge=None, progress=None):
        """titre, barre de progression, statistiques, pied."""
        rows = math.ceil(len(stats) / 2)
        height = 14 + 30 + (34 if progress else 0) + rows * 46 + (22 if footer else 0) + 6
        rounded_panel(self.screen, (x, y, width, height), PANEL, 12)

        cy = y + 14
        tx = x + 18
        if dot is not None:
            circle(self.screen, dot, (tx + 6, cy + 10), 6)
            tx += 20
        self.screen.blit(self.fonts["bold"].render(title, True, TEXT), (tx, cy))
        if badge is not None:
            image = self.fonts["small_bold"].render(badge, True, (29, 79, 134))
            bx = x + width - 18 - image.get_width() - 16
            rounded_panel(self.screen, (bx, cy - 1, image.get_width() + 16, 22), (227, 238, 249), 6)
            self.screen.blit(image, (bx + 8, cy + 2))
        cy += 30

        if progress:
            label, right, fraction = progress
            self.screen.blit(self.fonts["small"].render(label, True, MUTED), (x + 18, cy))
            image = self.fonts["small"].render(right, True, MUTED)
            self.screen.blit(image, (x + width - 18 - image.get_width(), cy))
            bar = pygame.Rect(x + 18, cy + 20, width - 36, 6)
            rounded_panel(self.screen, bar, (226, 230, 224), 3)
            rounded_panel(self.screen, (bar.x, bar.y, max(6, bar.width * fraction), 6), (46, 111, 184), 3)
            cy += 34

        column = (width - 36) / 2
        for i, (label, value) in enumerate(stats):
            sx = x + 18 + (i % 2) * column
            sy = cy + (i // 2) * 46
            self.screen.blit(self.fonts["small"].render(label, True, MUTED), (sx, sy))
            self.screen.blit(self.fonts["value"].render(value, True, TEXT), (sx, sy + 16))
        cy += rows * 46

        if footer:
            self.screen.blit(self.fonts["small"].render(footer, True, MUTED), (x + 18, cy + 2))
        return height

    def draw_keys(self, keys):
        """Rangée de pastilles en bas à gauche."""
        x, y = 20, self.height - 16 - 26
        for key, action in keys:
            key_image = self.fonts["small_bold"].render(key, True, TEXT)
            action_image = self.fonts["small"].render(" " + action, True, (51, 64, 58))
            w = key_image.get_width() + action_image.get_width() + 20
            rounded_panel(self.screen, (x, y, w, 26), (255, 255, 255, 210), 8)
            self.screen.blit(key_image, (x + 10, y + 5))
            self.screen.blit(action_image, (x + 10 + key_image.get_width(), y + 5))
            x += w + 8

    def draw_body_icon(self, spec, rect):
        """Petite silhouette d'un corps."""
        xs = [p[0] for s in spec.segments for p in (s.start, s.end)]
        ys = [p[1] for s in spec.segments for p in (s.start, s.end)]
        margin = max(s.width for s in spec.segments)
        span_x = max(xs) - min(xs) + margin
        span_y = max(ys) - min(ys) + margin
        rect = pygame.Rect(rect)
        scale = min(rect.width / span_x, rect.height / span_y)
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2

        def point(p):
            return (rect.centerx + (p[0] - cx) * scale, rect.centery - (p[1] - cy) * scale)

        outline = darken(spec.color, 0.68)
        for s in sorted(spec.segments, key=lambda s: s.width):
            capsule(self.screen, outline, point(s.start), point(s.end), s.width * scale + 3)
            capsule(self.screen, spec.color, point(s.start), point(s.end), s.width * scale)

    # ------------------------------------------------------------------
    def draw(self, space, text_lines=(), background=DEBUG_BACKGROUND, flip=True):
        """Dessin de débogage de pymunk + quelques lignes de texte en haut à gauche."""
        self.screen.fill(background)
        self._debug_space(space)
        for i, line in enumerate(text_lines):
            self.screen.blit(self.font.render(line, True, (40, 40, 40)), (10, 10 + 24 * i))
        if flip:
            pygame.display.flip()

    def handle_events(self):
        """Renvoie False si l'utilisateur ferme la fenêtre ou appuie sur Échap. D : débogage."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_d:
                    self.debug = not self.debug
        return True

    def flip(self):
        pygame.display.flip()

    def tick(self, fps):
        self.clock.tick(fps)

    def close(self):
        pygame.quit()

def fitness_text(value, unit):
    """Fitness à afficher avec son unité"""
    return format_number(value, 2, sign=True) + unit

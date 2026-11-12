import pygame
import pymunk
import pymunk.pygame_util

from simsengine.sim.world import GROUND_THICKNESS

class Renderer:
    """Affiche un espace pymunk dans une fenêtre pygame."""

    def __init__(self, width=1000, height=600, pixels_per_meter=100, title="SimsEngine") -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.width, self.height = width, height
        self.ppm = pixels_per_meter
        self.font = pygame.font.SysFont(None, 24)

        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        # Dessin des formes et des articulations
        self.draw_options.flags = (
            pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
            | pymunk.SpaceDebugDrawOptions.DRAW_CONSTRAINTS
        )
        self.center_on(0.0)

    def center_on(self, x, y=None):
        """Centre la caméra sur l'abscisse x."""
        self.camera_x = x
        ty = self.height - 50 if y is None else self.height / 2 + y * self.ppm
        self.draw_options.transform = pymunk.Transform(
            a=self.ppm, b=0, c=0, d=-self.ppm,
            tx=self.width / 2 - x * self.ppm,
            ty=ty,
        )

    def draw(self, space, text_lines=(), background=(245, 245, 240)):
        """Dessine l'espace, puis quelques lignes de texte en haut à gauche."""
        self.screen.fill(background)
        space.debug_draw(self.draw_options)
        self._draw_ground_marks()
        for i, line in enumerate(text_lines):
            image = self.font.render(line, True, (40, 40, 40))
            self.screen.blit(image, (10, 10 + 24 * i))
        pygame.display.flip()

    def _draw_ground_marks(self):
        """Dessin du repère sur le sol."""
        ground_y = self.height - 50
        below_ground = ground_y + GROUND_THICKNESS * self.ppm
        half_width_m = self.width / 2 / self.ppm
        first = int(self.camera_x - half_width_m) - 1
        last = int(self.camera_x + half_width_m) + 1
        for m in range(first, last + 1):
            sx = self.width / 2 + (m - self.camera_x) * self.ppm
            pygame.draw.line(self.screen, (90, 90, 90), (sx, ground_y), (sx, below_ground + 6), 2)
            label = self.font.render(f"{m} m", True, (150, 150, 150))
            self.screen.blit(label, (sx + 4, ground_y + 12))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def tick(self, fps):
        self.clock.tick(fps)

    def close(self):
        pygame.quit()

import pygame
import pymunk
import pymunk.pygame_util

class Renderer:
    """Affiche un espace pymunk dans une fenêtre pygame."""

    def __init__(self, width=1000, height=600, pixels_per_meter=100, title="SimsEngine") -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.width, self.height = width, height
        self.ppm = pixels_per_meter

        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        # Dessin des formes et des articulations
        self.draw_options.flags = (
            pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
            | pymunk.SpaceDebugDrawOptions.DRAW_CONSTRAINTS
        )
        self.center_on(0.0)

    def center_on(self, x):
        """Place le point (x, 0) du monde au centre horizontal"""
        self.draw_options.transform = pymunk.Transform(
            a=self.ppm, b=0, c=0, d=-self.ppm,
            tx=self.width / 2 - x * self.ppm,
            ty=self.height - 50,
        )

    def draw(self, space):
        self.screen.fill((245, 245, 240))
        space.debug_draw(self.draw_options)
        pygame.display.flip()

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

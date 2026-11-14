from pathlib import Path

import pygame

from simsengine.experiment import load_run
from simsengine.neat.network import FeedForwardNetwork
from simsengine.sim.body import load_body
from simsengine.sim.controllers import NeatController
from simsengine.sim.scenarios import SCENARIOS, get_scenario
from simsengine.sim.simulation import Simulation
from simsengine.viz.renderer import Renderer

BODIES_DIR = Path("bodies")
RESULTS_DIR = Path("results")
SLOW_MOTION = 4

TEXT = (40, 40, 40)
MUTED = (130, 130, 130)
HIGHLIGHT = (40, 110, 200)


def find_champion(body_name, scenario_name):
    """Renvoie le meilleur champion enregistré pour ce corps et ce scénario."""
    best = None
    for path in sorted(RESULTS_DIR.glob(f"{body_name}_{scenario_name}_seed*.json")):
        genome, history, info = load_run(path)
        if best is None or genome.fitness > best[1].fitness:
            best = (path, genome, history, info)
    return best


class Replay:
    """Rejoue un champion sur chaque épreuve de son scénario."""

    def __init__(self, genome, body, scenario):
        self.controller = NeatController(FeedForwardNetwork.create(genome))
        self.body = body
        self.variants = scenario.variants()
        self.variant_index = 0
        self.paused = False
        self.slow = False
        self.frame = 0
        self.restart()

    def restart(self):
        variant = self.variants[self.variant_index]
        self.sim = Simulation(self.controller, self.body, variant)

    def update(self):
        if self.paused:
            return
        self.frame += 1
        if self.slow and self.frame % SLOW_MOTION != 0:
            return
        self.sim.step()
        if self.sim.done:  # On passe à l'épreuve suivante
            self.variant_index = (self.variant_index + 1) % len(self.variants)
            self.restart()


class Demo:
    def __init__(self):
        self.renderer = Renderer(title="SimsEngine - démo")
        self.title_font = pygame.font.SysFont(None, 48)
        self.font = self.renderer.font
        self.body_files = sorted(BODIES_DIR.glob("*.yaml"))
        if not self.body_files:
            raise SystemExit(f"Aucun fichier de corps dans {BODIES_DIR}/")
        self.scenario_names = list(SCENARIOS)
        self.body_index = 0
        self.scenario_index = 0
        self.replay = None
        self._refresh_champion()

    @property
    def body_file(self):
        return self.body_files[self.body_index]

    @property
    def scenario_name(self):
        return self.scenario_names[self.scenario_index]

    def _refresh_champion(self):
        """Recherche le champion de la sélection (seulement quand elle change)."""
        self.champion = find_champion(self.body_file.stem, self.scenario_name)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if self.replay is None:
                        running = self._menu_key(event.key)
                    else:
                        self._replay_key(event.key)

            if self.replay is None:
                self._draw_menu()
            else:
                self.replay.update()
                self._draw_replay()
            self.renderer.tick(60)
        self.renderer.close()

    def _menu_key(self, key):
        """Gère une touche du menu. Renvoie False pour quitter la démo."""
        if key == pygame.K_ESCAPE:
            return False
        if key in (pygame.K_UP, pygame.K_DOWN):
            step = -1 if key == pygame.K_UP else 1
            self.body_index = (self.body_index + step) % len(self.body_files)
            self._refresh_champion()
        elif key in (pygame.K_LEFT, pygame.K_RIGHT):
            step = -1 if key == pygame.K_LEFT else 1
            self.scenario_index = (self.scenario_index + step) % len(self.scenario_names)
            self._refresh_champion()
        elif key == pygame.K_RETURN and self.champion is not None:
            _, genome, _, _ = self.champion
            self.replay = Replay(genome, load_body(self.body_file),
                                 get_scenario(self.scenario_name))
        return True

    def _draw_menu(self):
        screen = self.renderer.screen
        screen.fill((245, 245, 240))
        self._text("SimsEngine", (40, 30), self.title_font)

        # Colonne des créatures
        self._text("Créature (haut / bas)", (40, 100), color=MUTED)
        for i, path in enumerate(self.body_files):
            selected = i == self.body_index
            self._text(("> " if selected else "  ") + path.stem, (40, 130 + 28 * i),
                       color=HIGHLIGHT if selected else TEXT)

        # Colonne des scénarios
        self._text("Scénario (gauche / droite)", (360, 100), color=MUTED)
        for i, name in enumerate(self.scenario_names):
            selected = i == self.scenario_index
            self._text(("> " if selected else "  ") + name, (360, 130 + 28 * i),
                       color=HIGHLIGHT if selected else TEXT)

        # Champion de la sélection
        y = 380
        if self.champion is None:
            self._text("Aucun champion enregistré pour cette combinaison. Pour en créer un :",
                       (40, y))
            self._text(f"python -m scripts.run_evolution {self.body_file} "
                       f"--scenario {self.scenario_name}", (40, y + 28), color=MUTED)
        else:
            path, genome, history, info = self.champion
            self._text(f"Champion : fitness {genome.fitness:+.2f} après {len(history)} "
                       f"générations (graine {info['seed']})", (40, y))
            self._text(f"{path}", (40, y + 28), color=MUTED)
            self._text("Entrée : regarder le champion", (40, y + 70), color=HIGHLIGHT)
        self._text("Échap : quitter", (40, 560), color=MUTED)
        pygame.display.flip()

    def _replay_key(self, key):
        replay = self.replay
        assert replay is not None
        if key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
            self.replay = None
        elif key == pygame.K_SPACE:
            replay.paused = not replay.paused
        elif key == pygame.K_s:
            replay.slow = not replay.slow
        elif key == pygame.K_r:
            replay.variant_index = 0
            replay.restart()

    def _draw_replay(self):
        replay = self.replay
        assert replay is not None
        state = "PAUSE" if replay.paused else ("ralenti" if replay.slow else "")
        header = [
            f"{self.body_file.stem} - champion   {state}",
            f"épreuve {replay.variant_index + 1}/{len(replay.variants)}",
            "Espace : pause   S : ralenti   R : recommencer   Échap : menu",
            "",
        ]
        replay.sim.draw(self.renderer, header)

    def _text(self, text, position, font=None, color=TEXT):
        image = (font or self.font).render(text, True, color)
        self.renderer.screen.blit(image, position)

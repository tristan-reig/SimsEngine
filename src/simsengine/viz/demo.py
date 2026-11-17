from pathlib import Path

import pygame

from simsengine.experiment import load_run
from simsengine.live import LiveEvolution
from simsengine.neat.network import FeedForwardNetwork
from simsengine.sim.body import load_body
from simsengine.sim.controllers import NeatController
from simsengine.sim.scenarios import SCENARIOS, get_scenario
from simsengine.sim.simulation import Simulation
from simsengine.viz.drawing import format_number, rounded_panel
from simsengine.viz.renderer import MUTED, TEXT, Renderer, fitness_text

BODIES_DIR = Path("bodies")
RESULTS_DIR = Path("results")
SLOW_MOTION = 4
LIVE_GENERATIONS = 50
TOAST_FRAMES = 180

MENU_BACKGROUND = (244, 246, 241)
CARD = (255, 255, 255)
BLUE = (46, 111, 184)
ORANGE = (217, 119, 42)
LINE = (229, 232, 227)


def find_champion(body_name, scenario_name):
    """Le meilleur champion enregistré pour ce corps et ce scénario."""
    best = None
    for path in sorted(RESULTS_DIR.glob(f"{body_name}_{scenario_name}_seed*.json")):
        genome, history, info = load_run(path)
        if best is None or genome.fitness > best[1].fitness:
            best = (path, genome, history, info)
    return best


class Replay:
    """Rejoue un génome sur chaque épreuve d'un scénario."""

    def __init__(self, get_genome, body, scenario):
        self.get_genome = get_genome
        self.body = body
        self.scenario = scenario
        self.variants = scenario.variants()
        self.variant_index = 0
        self.genome = None
        self.sim = None
        self.paused = False
        self.slow = False
        self.frame = 0
        self.champion_changed = False  # passe à True quand un nouveau champion arrive
        self.restart()

    def restart(self):
        genome = self.get_genome()
        if genome is None:
            return
        if genome is not self.genome:
            self.champion_changed = self.genome is not None
            self.genome = genome
            self.controller = NeatController(FeedForwardNetwork.create(genome))
            self.variant_index = 0
        self.sim = Simulation(self.controller, self.body, self.variants[self.variant_index])

    def update(self):
        """Fait avancer la simulation."""
        if self.sim is None:
            self.restart()
            return
        if self.paused:
            return
        self.frame += 1
        if self.slow and self.frame % SLOW_MOTION != 0:
            return
        self.sim.step()
        if self.sim.done:
            self.variant_index = (self.variant_index + 1) % len(self.variants)
            self.restart()


class Demo:
    def __init__(self):
        self.renderer = Renderer(title="SimsEngine - démo")
        self.fonts = self.renderer.fonts
        self.body_files = sorted(BODIES_DIR.glob("*.yaml"))
        if not self.body_files:
            raise SystemExit(f"Aucun fichier de corps dans {BODIES_DIR}/")
        self.bodies = [load_body(path) for path in self.body_files]
        self.scenario_names = list(SCENARIOS)
        self.scenarios = [get_scenario(name) for name in self.scenario_names]
        self.body_index = 0
        self.scenario_index = 0
        self.replay = None
        self.live = None
        self.toast = None
        self._refresh_champion()

    @property
    def body_file(self):
        return self.body_files[self.body_index]

    @property
    def body(self):
        return self.bodies[self.body_index]

    @property
    def scenario(self):
        return self.scenarios[self.scenario_index]

    def _refresh_champion(self):
        """Recherche le champion de la sélection."""
        self.champion = find_champion(self.body_file.stem, self.scenario.name)

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

        if self.live is not None:
            self.live.stop()
        self.renderer.close()

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------
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
            self.scenario_index = (self.scenario_index + step) % len(self.scenarios)
            self._refresh_champion()
        elif key == pygame.K_RETURN and self.champion is not None:
            genome = self.champion[1]
            self.replay = Replay(lambda: genome, self.body, get_scenario(self.scenario.name))
        elif key == pygame.K_e:
            live = LiveEvolution(self.body_file, self.scenario.name, LIVE_GENERATIONS)
            self.live = live
            self.replay = Replay(
                lambda: live.snapshot()[0], self.body, get_scenario(self.scenario.name)
            )
        return True

    def _draw_menu(self):
        screen = self.renderer.screen
        screen.fill(MENU_BACKGROUND)
        self._text("SimsEngine", (48, 36), "title")
        self._text("Créatures évolutives - NEAT et simulation physique", (48, 84), "body", MUTED)

        # Colonne des créatures
        self._label("CRÉATURE · ↑ ↓", (48, 130))
        for i, body in enumerate(self.bodies):
            rect = pygame.Rect(48, 156 + i * 72, 300, 62)
            self._card(rect, selected=i == self.body_index)
            self.renderer.draw_body_icon(body, (rect.x + 12, rect.y + 10, 60, 42))
            self._text(body.name, (rect.x + 86, rect.y + 12), "bold")
            self._text(
                f"{len(body.segments)} segments · {len(body.joints)} articulations",
                (rect.x + 86, rect.y + 34),
                "small",
                MUTED,
            )

        # Colonne des scénarios
        self._label("SCÉNARIO · ← →", (376, 130))
        for i, scenario in enumerate(self.scenarios):
            rect = pygame.Rect(376, 156 + i * 58, 220, 48)
            selected = i == self.scenario_index
            self._card(rect, selected)
            self._text(scenario.title, (rect.x + 16, rect.y + 14), "bold" if selected else "body")

        self._draw_champion_card(pygame.Rect(624, 130, 328, 380))
        x = 48
        for key, action in (("↑ ↓", "créature"), ("← →", "scénario"), ("Échap", "quitter")):
            image = self.fonts["small_bold"].render(key, True, (51, 64, 58))
            screen.blit(image, (x, 556))
            x += image.get_width()
            image = self.fonts["small"].render(" " + action, True, MUTED)
            screen.blit(image, (x, 556))
            x += image.get_width() + 22
        pygame.display.flip()

    def _draw_champion_card(self, rect):
        rounded_panel(self.renderer.screen, rect, CARD, 16)
        x, y = rect.x + 22, rect.y + 22
        self._label("CHAMPION ENREGISTRÉ", (x, y))
        if self.champion is None:
            self._text("Aucun champion pour", (x, y + 34), "bold")
            self._text("cette combinaison.", (x, y + 56), "bold")
            self._text("Lance une évolution en direct :", (x, y + 92), "small", MUTED)
            self._text("elle sera enregistrée à la fin.", (x, y + 110), "small", MUTED)
        else:
            path, genome, history, info = self.champion
            self._text(fitness_text(genome.fitness, self.scenario.unit), (x, y + 28), "title")
            species = history[-1].get("species", 1)
            self._text(
                f"{len(history)} générations · graine {info['seed']} · "
                f"{species} espèce{'s' if species > 1 else ''}",
                (x, y + 78),
                "small",
                MUTED,
            )
            self._sparkline(
                [h["best"] for h in history], pygame.Rect(x, y + 104, rect.width - 44, 44)
            )
            self._text(str(path), (x, rect.bottom - 34), "small", MUTED)
        pygame.draw.line(self.renderer.screen, LINE, (x, y + 166), (rect.right - 22, y + 166))

        buttons = [("E", "Évolution en direct", f"{LIVE_GENERATIONS} gén.")]
        if self.champion is not None:
            buttons.insert(0, ("Entrée", "Regarder le champion", ""))
        for i, (key, label, right) in enumerate(buttons):
            button = pygame.Rect(x, y + 184 + i * 58, rect.width - 44, 46)
            primary = i == 0 and self.champion is not None
            if primary:
                rounded_panel(self.renderer.screen, button, BLUE, 10)
            else:
                rounded_panel(self.renderer.screen, button, (201, 211, 221), 10)
                rounded_panel(self.renderer.screen, button.inflate(-3, -3), CARD, 9)
            text_color = (255, 255, 255) if primary else TEXT
            key_image = self.fonts["small_bold"].render(key, True, text_color)
            chip = pygame.Rect(button.x + 12, button.y + 12, key_image.get_width() + 14, 22)
            rounded_panel(
                self.renderer.screen, chip, (255, 255, 255, 55) if primary else (238, 241, 236), 6
            )
            self.renderer.screen.blit(key_image, (chip.x + 7, chip.y + 3))
            self._text(label, (chip.right + 10, button.y + 13), "bold", text_color)
            if right:
                image = self.fonts["small"].render(right, True, MUTED)
                self.renderer.screen.blit(
                    image, (button.right - 12 - image.get_width(), button.y + 15)
                )

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------
    def _replay_key(self, key):
        replay = self.replay
        assert replay is not None
        if key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
            self._back_to_menu()
        elif key == pygame.K_SPACE:
            replay.paused = not replay.paused
        elif key == pygame.K_s:
            replay.slow = not replay.slow
        elif key == pygame.K_d:
            self.renderer.debug = not self.renderer.debug
        elif key == pygame.K_r:
            replay.variant_index = 0
            replay.restart()

    def _back_to_menu(self):
        if self.live is not None:
            self._message_screen("Arrêt de l'évolution et enregistrement du champion...")
            self.live.stop()
            self.live = None
        self.replay = None
        self.toast = None
        self._refresh_champion()  # un nouveau champion a peut-être été enregistré

    def _draw_replay(self):
        replay = self.replay
        assert replay is not None
        if replay.sim is None or replay.genome is None:
            self._message_screen("Évaluation de la génération 0 en cours...")
            return

        sim = replay.sim
        self.renderer.draw_world(sim)
        state = "pause" if replay.paused else ("ralenti" if replay.slow else None)
        episode = (
            f"{self.scenario.title} · épreuve {replay.variant_index + 1}/{len(replay.variants)}"
        )

        if self.live is None:
            self.renderer.draw_card(
                20,
                20,
                270,
                f"{self.body.name} · champion",
                sim.hud_stats(),
                footer=episode,
                dot=self.body.color,
                badge=state,
            )
            keys = [
                ("Espace", "pause"),
                ("S", "ralenti"),
                ("R", "recommencer"),
                ("D", "vue débogage"),
                ("Échap", "menu"),
            ]
        else:
            self._draw_live_panels(replay, sim, state, episode)
            keys = [
                ("Espace", "pause"),
                ("S", "ralenti"),
                ("D", "vue débogage"),
                ("Échap", "arrêter et enregistrer"),
            ]
        self.renderer.draw_keys(keys)
        pygame.display.flip()

    def _draw_live_panels(self, replay, sim, state, episode):
        live = self.live
        assert live is not None and replay.genome is not None
        _, history = live.snapshot()
        generation = len(history)
        badge = state or ("terminée" if live.finished else "en cours")
        species = history[-1]["species"] if history else 1
        threshold = format_number(history[-1]["threshold"], 2) if history else "–"
        stats = [
            ("Individu affiché", fitness_text(replay.genome.fitness, self.scenario.unit)),
            ("Espèces", f"{species}  (seuil {threshold})"),
        ]
        self.renderer.draw_card(
            20,
            20,
            300,
            "Évolution en direct",
            stats,
            footer=f"{self.body.name} · {episode}",
            badge=badge,
            progress=(
                "Génération",
                f"{generation} / {LIVE_GENERATIONS}",
                generation / LIVE_GENERATIONS,
            ),
        )
        self._draw_curve(history)

        # Bandeau "nouveau champion"
        if replay.champion_changed:
            replay.champion_changed = False
            found = next(
                (h["generation"] for h in history if h["best"] >= replay.genome.fitness - 1e-9),
                generation - 1,
            )
            text = (
                f"Nouveau champion à la génération {found} : "
                f"{fitness_text(replay.genome.fitness, self.scenario.unit)}"
            )
            self.toast = (text, TOAST_FRAMES)
        if self.toast is not None:
            text, frames = self.toast
            image = self.fonts["body"].render(text, True, (255, 255, 255))
            rect = pygame.Rect(0, 0, image.get_width() + 28, 36)
            rect.center = (self.renderer.width // 2, 232)
            rounded_panel(self.renderer.screen, rect, (31, 42, 36, 235), 18)
            self.renderer.screen.blit(image, (rect.x + 14, rect.y + 8))
            self.toast = (text, frames - 1) if frames > 1 else None

    def _draw_curve(self, history):
        """Courbe de fitness en haut à droite."""
        rect = pygame.Rect(self.renderer.width - 340, 20, 320, 176)
        rounded_panel(self.renderer.screen, rect, (255, 255, 255, 235), 12)
        self._text("- meilleure", (rect.x + 16, rect.y + 12), "small_bold", BLUE)
        self._text("- moyenne", (rect.x + 112, rect.y + 12), "small_bold", ORANGE)
        plot = pygame.Rect(rect.x + 20, rect.y + 44, rect.width - 40, 104)
        for i in range(3):
            y = plot.y + plot.height * i / 2
            pygame.draw.line(self.renderer.screen, LINE, (plot.x, y), (plot.right, y))
        if len(history) < 2:
            self._text("courbe après 2 générations", (plot.x, plot.centery - 8), "small", MUTED)
            return
        values = [h["best"] for h in history] + [h["mean"] for h in history]
        low, high = min(values), max(values)
        span = high - low or 1.0

        def point(i, value):
            return (
                plot.x + plot.width * i / (len(history) - 1),
                plot.bottom - plot.height * (value - low) / span,
            )

        for key, color in (("mean", ORANGE), ("best", BLUE)):
            points = [point(i, h[key]) for i, h in enumerate(history)]
            pygame.draw.lines(self.renderer.screen, color, False, points, 3)
        self._text(format_number(high, 1), (rect.right - 50, rect.y + 12), "small", MUTED)
        self._text(f"gén. {len(history) - 1}", (rect.right - 60, rect.bottom - 22), "small", MUTED)
        self._text("gén. 0", (plot.x, rect.bottom - 22), "small", MUTED)

    # ------------------------------------------------------------------
    # Outils de dessin
    # ------------------------------------------------------------------
    def _sparkline(self, values, rect):
        if len(values) < 2:
            return
        low, high = min(values), max(values)
        span = high - low or 1.0
        points = [
            (
                rect.x + rect.width * i / (len(values) - 1),
                rect.bottom - rect.height * (v - low) / span,
            )
            for i, v in enumerate(values)
        ]
        pygame.draw.lines(self.renderer.screen, BLUE, False, points, 3)

    def _card(self, rect, selected):
        if selected:
            rounded_panel(self.renderer.screen, rect, BLUE, 12)
            rounded_panel(self.renderer.screen, rect.inflate(-4, -4), CARD, 10)
        else:
            rounded_panel(self.renderer.screen, rect, CARD, 12)

    def _label(self, text, position):
        self._text(text, position, "small_bold", MUTED)

    def _message_screen(self, message):
        self.renderer.screen.fill(MENU_BACKGROUND)
        self._text(message, (48, 280), "bold")
        pygame.display.flip()

    def _text(self, text, position, font="body", color=TEXT):
        self.renderer.screen.blit(self.fonts[font].render(text, True, color), position)

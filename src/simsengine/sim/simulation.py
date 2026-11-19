from .creature import Creature
from .scenarios import WalkScenario
from .world import DT, SUBSTEPS


def physics_step(space, creature, scenario):
    """Avance la physique de DT, découpé en petits pas."""
    for _ in range(SUBSTEPS):
        scenario.apply_forces(creature)  # les forces sont remises à zéro à chaque pas
        space.step(DT / SUBSTEPS)


class Simulation:
    """Une créature pilotée par un contrôleur dans un scénario, pas à pas."""

    def __init__(self, controller, body, scenario=None, duration=10.0):
        self.controller = controller
        self.scenario = scenario if scenario is not None else WalkScenario()
        self.duration = duration
        self.space = self.scenario.create_world()
        self.creature = Creature(self.space, body)
        self.scenario.setup(self.creature)
        self.start_x = self.creature.position_x
        self.n_steps = int(duration / DT)
        self.step_count = 0

    @property
    def t(self):
        """Temps écoulé."""
        return self.step_count * DT

    @property
    def done(self):
        """Fin de l'épreuve : durée écoulée, ou scénario terminé."""
        return self.step_count >= self.n_steps or self.scenario.finished(self.creature)

    def step(self):
        """Avance la simulation de DT."""
        creature, scenario = self.creature, self.scenario
        observation = creature.observe() + scenario.extra_inputs(creature)
        creature.set_commands(self.controller.act(observation, self.t))
        physics_step(self.space, creature, scenario)
        scenario.update(creature)
        self.step_count += 1

    def fitness(self):
        return self.scenario.fitness(self.creature, self.start_x)

    def hud_stats(self):
        """Informations à afficher : le temps, puis celles du scénario."""
        time = f"{self.t:.1f} s".replace(".", ",")
        return [("Temps", time)] + self.scenario.hud_stats(self.creature, self.start_x)

    def draw(self, renderer, title=None, footer=None, keys=(), flip=True):
        """Dessine la simulation et une carte d'informations en haut à gauche."""
        renderer.draw_world(self)
        renderer.draw_card(
            20,
            20,
            270,
            title or self.creature.spec.name,
            self.hud_stats(),
            footer=footer or self.scenario.title,
            dot=self.creature.spec.color,
        )
        if keys:
            renderer.draw_keys(keys)
        if flip:
            renderer.flip()


def simulate(controller, body, scenario=None, duration=10.0, renderer=None):
    """Simule une créature pilotée par `controller` dans un scénario."""
    sim = Simulation(controller, body, scenario, duration)
    while not sim.done:
        sim.step()
        if renderer is not None:
            if not renderer.handle_events():
                break
            sim.draw(renderer, keys=[("D", "vue débogage"), ("Échap", "quitter")])
            renderer.tick(60)
    return sim.fitness()

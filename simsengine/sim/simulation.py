from .creature import Creature
from .scenarios import WalkScenario

DT = 1.0 / 60.0
SUBSTEPS = 4

def physics_step(space, creature, scenario):
    """Avance la physique de DT, découpé en SUBSTEPS petits pas."""
    for _ in range(SUBSTEPS):
        scenario.apply_forces(creature)
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
        return self.step_count >= self.n_steps

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

    def info_lines(self):
        lines = [f"scénario : {self.scenario.name}", f"temps : {self.t:5.1f} s"]
        return lines + self.scenario.info_lines(self.creature, self.start_x)

    def draw(self, renderer, extra_lines=()):
        """Dessine la simulation avec la caméra centrée sur la créature."""
        # Sans sol, la caméra suit aussi la créature verticalement
        follow_y = None if self.scenario.has_ground else self.creature.position_y
        renderer.center_on(self.creature.position_x, follow_y)
        renderer.draw(self.space, list(extra_lines) + self.info_lines(),
                      background=self.scenario.background)


def simulate(controller, body, scenario=None, duration=10.0, renderer=None):
    """Simule une créature pilotée par `controller` dans un scénario."""

    sim = Simulation(controller, body, scenario, duration)
    while not sim.done:
        sim.step()
        if renderer is not None:
            if not renderer.handle_events():
                break
            sim.draw(renderer)
            renderer.tick(60)
    return sim.fitness()

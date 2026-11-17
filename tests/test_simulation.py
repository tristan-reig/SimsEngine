import math
from pathlib import Path

import pytest
from pymunk.constraints import PivotJoint

from simsengine.evaluation import Evaluator, evaluate_genome
from simsengine.neat.config import NEATConfig
from simsengine.neat.population import Population
from simsengine.sim.body import load_body
from simsengine.sim.controllers import SineController, neat_sizes
from simsengine.sim.creature import Creature
from simsengine.sim.scenarios import SCENARIOS, SwimScenario, get_scenario
from simsengine.sim.simulation import Simulation, physics_step, simulate
from simsengine.sim.world import add_ball, add_box, create_world

BODIES = sorted(Path("bodies").glob("*.yaml"))
WORM = Path("bodies/worm.yaml")


def test_falling_objects_rest_on_the_ground():
    space = create_world()
    bodies = [add_box(space, (0.0, 2.0), (0.4, 0.3)), add_ball(space, (1.0, 3.0), 0.2)]
    for _ in range(600):
        space.step(1 / 60)
    for body in bodies:
        assert 0.0 < body.position.y < 0.5
        assert body.velocity.length < 0.05


@pytest.mark.parametrize("path", BODIES, ids=lambda p: p.stem)
def test_every_body_stands_still_when_holding_its_pose(path):
    scenario = get_scenario("walk")
    space = scenario.create_world()
    creature = Creature(space, load_body(path))
    start = creature.position_x
    for _ in range(600):
        creature.set_targets([0.0] * len(creature.joints))
        physics_step(space, creature, scenario)
    assert abs(creature.position_x - start) < 0.02


def test_unknown_segment_in_a_joint_is_an_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "defaults: {segment: {width: 0.1, mass: 1, friction: 1},"
        " joint: {min_angle: -45, max_angle: 45, max_torque: 10, max_speed: 500, gain: 10}}\n"
        "segments: [{name: a, from: [0, 0.1], to: [1, 0.1]}]\n"
        "joints: [{between: [a, b], at: [0.5, 0.1]}]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inconnu"):
        load_body(bad)


def test_travelling_wave_moves_the_worm_forward_and_backward():
    body = load_body(WORM)
    forward = simulate(SineController(3, 0.75, 1.0, math.radians(90)), body)
    backward = simulate(SineController(3, 0.75, 1.0, math.radians(-90)), body)
    assert forward > 1.0
    assert backward < -1.0


def test_simulation_is_deterministic():
    body = load_body(WORM)
    controller = SineController(3, 0.75, 1.0, math.radians(90))
    assert simulate(controller, body, SwimScenario()) == simulate(controller, body, SwimScenario())


def test_joints_stay_attached_under_brutal_commands():
    body = load_body(WORM)

    class BangBang:
        def act(self, observation, t):
            return [1.0 if int(t * 4) % 2 else -1.0] * 3

    sim = Simulation(BangBang(), body)
    worst = 0.0
    while not sim.done:
        sim.step()
        for c in sim.space.constraints:
            if isinstance(c, PivotJoint):
                gap = (c.a.local_to_world(c.anchor_a) - c.b.local_to_world(c.anchor_b)).length
                worst = max(worst, gap)
    assert worst < 0.02


@pytest.mark.parametrize("name", SCENARIOS)
def test_network_inputs_match_observations(name):
    body = load_body(WORM)
    scenario = get_scenario(name)
    n_inputs, _ = neat_sizes(body, scenario)
    sim = Simulation(SineController(3, 0.5, 1.0, 1.0), body, scenario)
    observation = sim.creature.observe() + scenario.extra_inputs(sim.creature)
    assert len(observation) + 2 == n_inputs


def test_parallel_evaluation_gives_the_same_fitnesses():
    body, scenario = load_body(WORM), get_scenario("food")
    n_inputs, n_outputs = neat_sizes(body, scenario)
    genomes = Population(NEATConfig(), n_inputs, n_outputs, seed=0).genomes[:8]
    sequential = [evaluate_genome(g, body, scenario) for g in genomes]
    with Evaluator(body, scenario, workers=2) as evaluate:
        assert evaluate(genomes) == sequential

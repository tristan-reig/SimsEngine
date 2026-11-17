import statistics
import sys

from simsengine.neat.config import NEATConfig
from simsengine.neat.network import FeedForwardNetwork
from simsengine.neat.population import Population

XOR_CASES = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]


def fitness(genome):
    net = FeedForwardNetwork.create(genome)
    error = sum(abs(net.activate(x)[0] - y) for x, y in XOR_CASES)
    return (4 - error) ** 2


def evaluate(genomes):
    return [fitness(g) for g in genomes]


def solves_xor(genome):
    net = FeedForwardNetwork.create(genome)
    return all((net.activate(x)[0] > 0.5) == (y == 1) for x, y in XOR_CASES)


def count_hidden(genome):
    return sum(1 for n in genome.nodes.values() if n.type.value == "hidden")


config = NEATConfig()
n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1

if n_runs == 1:
    pop = Population(config, n_inputs=2, n_outputs=1, seed=0)
    best = pop.run(evaluate, max_generations=300, stop_when=solves_xor)

    print("\nRésolu !" if solves_xor(best) else "\nNon résolu en 300 générations.")
    print(best.describe())
    net = FeedForwardNetwork.create(best)
    for x, y in XOR_CASES:
        print(f"  {x} -> {net.activate(x)[0]:.3f}   (attendu {y})")
else:
    generations, hidden, failures = [], [], 0
    for seed in range(n_runs):
        pop = Population(config, n_inputs=2, n_outputs=1, seed=seed)
        best = pop.run(evaluate, max_generations=300, stop_when=solves_xor, verbose=False)
        if solves_xor(best):
            generations.append(pop.generation)
            hidden.append(count_hidden(best))
            print(
                f"Run {seed + 1:3d} : résolu en {pop.generation:3d} générations, "
                f"{count_hidden(best)} noeuds cachés"
            )
        else:
            failures += 1
            print(f"Run {seed + 1:3d} : échec")

    print(f"\n{n_runs - failures}/{n_runs} runs réussis")
    if generations:
        print(
            f"Générations : moyenne {statistics.mean(generations):.1f}, "
            f"médiane {statistics.median(generations):.0f}"
        )
        print(f"Noeuds cachés : moyenne {statistics.mean(hidden):.2f}")

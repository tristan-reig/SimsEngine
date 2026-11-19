import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from simsengine.experiment import load_run
from simsengine.sim.scenarios import get_scenario

RESULTS_DIR = Path("results") / "experiments"
COLORS = {"adaptive": "#2a78d6", "fixed": "#eb6834", "early": "#1baf7a", "sine": "#eda100"}
LABELS = {
    "adaptive": "seuil adaptatif, départ 3,0",
    "fixed": "seuil fixe 3,0 (une espèce)",
    "early": "seuil adaptatif, départ 1,0",
    "sine": "sinus aléatoire (sans NEAT)",
}
REFERENCE = "fixed"


def load_groups():
    """Regroupe les historiques par (créature, scénario, condition)."""
    groups = defaultdict(dict)
    for path in sorted(RESULTS_DIR.glob("*.json")):
        _, history, info = load_run(path)
        key = (info["body"], info["scenario"], info["condition"])
        groups[key][info["seed"]] = history
    return groups


def curves(runs, key):
    """Tableau numpy (graines x générations) de la statistique `key`."""
    length = min(len(history) for history in runs.values())
    return np.array([[h[key] for h in history[:length]] for history in runs.values()])


def has_species(runs):
    """Les historiques NEAT comptent les espèces, pas ceux de la référence sinus."""
    return all("species" in history[0] for history in runs.values())


def species_curves(runs):
    return curves(runs, "species") if has_species(runs) else None


def best_so_far(runs):
    """Meilleure fitness trouvée jusqu'ici."""
    return np.maximum.accumulate(curves(runs, "best"), axis=1)


def std(values, axis=None):
    """Écart-type de l'échantillon, sur les graines."""
    if np.shape(values)[0] < 2:
        return np.zeros_like(np.mean(values, axis=axis))
    return np.std(values, axis=axis, ddof=1)


def plot_grid(groups, path, statistic, ylabel, integer=False):
    """Une case par (créature, scénario), une courbe moyenne + ou - écart-type par condition."""
    bodies = sorted({body for body, _, _ in groups})
    scenarios = sorted({scenario for _, scenario, _ in groups})
    conditions = sorted({condition for _, _, condition in groups})
    fig, axes = plt.subplots(
        len(bodies),
        len(scenarios),
        figsize=(4.2 * len(scenarios), 3.2 * len(bodies)),
        squeeze=False,
    )
    for row, body in enumerate(bodies):
        for col, scenario_name in enumerate(scenarios):
            ax = axes[row][col]
            for condition in conditions:
                runs = groups.get((body, scenario_name, condition))
                values = statistic(runs) if runs else None
                if values is None:
                    continue
                mean, spread = values.mean(axis=0), std(values, axis=0)
                generations = np.arange(len(mean))
                color = COLORS.get(condition)
                ax.plot(
                    generations,
                    mean,
                    color=color,
                    linewidth=2,
                    label=f"{LABELS.get(condition, condition)} ({len(runs)} graines)",
                )
                ax.fill_between(generations, mean - spread, mean + spread, color=color, alpha=0.15)
            ax.set_title(f"{body} · {get_scenario(scenario_name).title}")
            ax.grid(alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            if integer:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if row == len(bodies) - 1:
                ax.set_xlabel("génération")
            if col == 0:
                ax.set_ylabel(ylabel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    columns = min(len(handles), 2)
    fig.legend(handles, labels, loc="upper center", ncol=columns, frameon=False)
    legend_rows = math.ceil(len(handles) / columns)
    top = 1 - (0.15 + 0.3 * legend_rows) / fig.get_figheight()
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary_rows(groups):
    """Une ligne de tableau par (créature, scénario, condition)."""
    rows = []
    for (body, scenario, condition), runs in sorted(groups.items()):
        final = best_so_far(runs)[:, -1]  # fitness du champion de chaque graine
        species = curves(runs, "species")[:, -1] if has_species(runs) else None
        rows.append(
            {
                "creature": body,
                "scenario": scenario,
                "condition": condition,
                "graines": len(runs),
                "generations": curves(runs, "best").shape[1],
                "fitness_moyenne": round(float(final.mean()), 3),
                "fitness_ecart_type": round(float(std(final)), 3),
                "fitness_min": round(float(final.min()), 3),
                "fitness_max": round(float(final.max()), 3),
                "especes_finales": "" if species is None else round(float(species.mean()), 1),
            }
        )
    return rows


def compare(groups, a, b):
    """Compare deux conditions graine par graine."""
    print(f"\n{LABELS.get(a, a)}  comparé à  {LABELS.get(b, b)}, graine par graine :")
    for body, scenario, condition in sorted(groups):
        if condition != a or (body, scenario, b) not in groups:
            continue
        runs_a, runs_b = groups[(body, scenario, a)], groups[(body, scenario, b)]
        seeds = sorted(runs_a.keys() & runs_b.keys())  # graines présentes des deux côtés
        final_a = best_so_far({s: runs_a[s] for s in seeds})[:, -1]
        final_b = best_so_far({s: runs_b[s] for s in seeds})[:, -1]
        difference = final_a - final_b
        wins = int((difference > 1e-9).sum())
        losses = int((difference < -1e-9).sum())
        print(
            f"  {body:10s} {scenario:6s} : écart moyen {difference.mean():+6.2f} "
            f"(± {std(difference):.2f}) | meilleur sur {wins}/{len(seeds)}, "
            f"moins bon sur {losses}/{len(seeds)}"
        )


def main():
    groups = load_groups()
    if not groups:
        raise SystemExit(f"Aucune évolution dans {RESULTS_DIR}/ : lance d'abord make experiments")

    rows = summary_rows(groups)
    csv_path = RESULTS_DIR / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'créature':10s} {'scénario':8s} {'condition':9s}  fitness finale (moy. ± é.-t.)")
    for r in rows:
        species = r["especes_finales"]
        print(
            f"{r['creature']:10s} {r['scenario']:8s} {r['condition']:9s}  "
            f"{r['fitness_moyenne']:7.2f} ± {r['fitness_ecart_type']:5.2f}   "
            f"[{r['fitness_min']:.2f} ; {r['fitness_max']:.2f}]   "
            f"{'' if species == '' else f'{species:.1f} espèces'}"
        )
    for condition in sorted({condition for _, _, condition in groups} - {REFERENCE}):
        compare(groups, condition, REFERENCE)

    plot_grid(groups, RESULTS_DIR / "fitness.png", best_so_far, "meilleure fitness")
    plot_grid(
        groups,
        RESULTS_DIR / "species.png",
        species_curves,
        "nombre d'espèces",
        integer=True,
    )
    print(f"\nÉcrit : {csv_path}, {RESULTS_DIR / 'fitness.png'}, {RESULTS_DIR / 'species.png'}")


if __name__ == "__main__":
    main()

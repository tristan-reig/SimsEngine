import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(history, path, title=""):
    """Trace l'évolution de la fitness et du nombre d'espèces, en sauvegardant."""
    generations = [h["generation"] for h in history]
    fig, (ax_fit, ax_species) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                             gridspec_kw={"height_ratios": [3, 1]})

    ax_fit.plot(generations, [h["best"] for h in history], label="meilleure")
    ax_fit.plot(generations, [h["mean"] for h in history], label="moyenne")
    ax_fit.set_ylabel("fitness")
    ax_fit.legend()
    ax_fit.grid(alpha=0.3)

    ax_species.plot(generations, [h["species"] for h in history], color="gray")
    ax_species.set_ylabel("espèces")
    ax_species.set_xlabel("génération")
    ax_species.grid(alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

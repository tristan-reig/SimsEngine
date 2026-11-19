import json
from pathlib import Path

from simsengine.neat.genome import Genome


def save_run(path, best_genome, history, info):
    """Enregistre une expérience dans un fichier JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # crée le dossier results/ si besoin
    data = {"info": info, "history": history}
    if best_genome is not None:
        data["best_genome"] = best_genome.to_dict()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_run(path):
    """Relit un fichier écrit par `save_run`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    genome = Genome.from_dict(data["best_genome"]) if "best_genome" in data else None
    return genome, data["history"], data["info"]

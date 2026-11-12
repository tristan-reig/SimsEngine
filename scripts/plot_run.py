import sys
from pathlib import Path

from simsengine.experiment import load_run
from simsengine.viz.plots import plot_history

path = Path(sys.argv[1])
genome, history, info = load_run(path)
image = path.with_suffix(".png")
plot_history(history, image, title=f"{info['body']}, graine {info['seed']}")
print(f"Courbes enregistrées dans {image}")

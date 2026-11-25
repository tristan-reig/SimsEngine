# SimsEngine

Des créatures virtuelles qui apprennent à se déplacer toutes seules.

Chaque créature est un corps articulé, simulé en 2D (vue de profil) avec le moteur physique
[pymunk](https://www.pymunk.org). Son "cerveau" est un réseau de neurones qui lit ses capteurs
et commande ses articulations. Personne n'écrit ce réseau à la main : il est trouvé par
**NEAT** (*NeuroEvolution of Augmenting Topologies*, Stanley et Miikkulainen, 2002), un
algorithme évolutionnaire qui fait évoluer à la fois les poids et la structure des réseaux.

## Ce que contient le projet

- un **moteur générique** : n'importe quel corps décrit dans un fichier YAML peut être simulé et
  entraîné, sans écrire de code ;
- une **implémentation de NEAT** écrite de zéro (innovations, spéciation, croisement...) ;
- **5 créatures** et **5 scénarios**, soit 25 combinaisons ;
- une **démo interactive** pour regarder les champions, ou une évolution en direct ;
- des **scripts d'expériences** qui comparent plusieurs réglages sur plusieurs graines, avec
  courbes moyennes et tableau récapitulatif.

| Créatures | | Scénarios | |
|---|---|---|---|
| Ver | 4 segments à plat | Marche | aller le plus loin possible vers la droite |
| Serpent | 8 segments, 7 articulations | Nage | avancer dans l'eau, sans sol |
| Quadrupède | un tronc, une patte avant et une patte arrière (de profil) | Nourriture | manger des boules, à gauche puis à droite |
| Sauteur | une jambe et une queue d'appui | Ballon | pousser un ballon du côté où il se trouve |
| Bipède | un tronc et 2 jambes | Rocher | fuir un rocher qui accélère |

## Installation

Il faut Python 3.11 ou plus récent. Le projet a été développé sous Linux (WSL). Pour les
fenêtres graphiques sous WSL, il faut WSLg (inclus dans Windows 11).

```bash
git clone https://github.com/tristan24217/SimsEngine.git SimsEngine
cd SimsEngine
python3 -m venv .venv
source .venv/bin/activate
make install
make check            # vérifie que tout fonctionne
```

## Premiers pas

```bash
make demo                                                  # la démo interactive
python -m scripts.run_evolution bodies/worm.yaml           # faire évoluer le ver
python -m scripts.replay results/worm_walk_seed0.json      # regarder le champion
```

Dans la démo, tout se fait au clavier ou à la souris : choisir une créature et un scénario,
puis regarder le champion enregistré (**Entrée**) ou lancer une évolution en direct (**E**).
Pendant un replay : **Espace** pour la pause, **S** pour le ralenti, **R** pour recommencer,
**D** pour la vue de débogage (formes physiques réelles), **Échap** pour revenir au menu.

## Commandes

`make help` affiche la liste.

| Commande | Rôle |
|---|---|
| `make install` | installe le projet et les outils de développement |
| `make check` | lint, types et tests : à lancer avant chaque commit |
| `make lint` / `make format` | vérifie / corrige le style (ruff) |
| `make typecheck` | vérifie les types (pyright) |
| `make test` | lance les tests (pytest) |
| `make demo` | lance la démo interactive |
| `make xor` | valide NEAT sur XOR (20 évolutions) |
| `make experiments` | NEAT contre sinus aléatoire, sur les 25 combinaisons (long) |
| `make speciation` | étude de la spéciation (long) |
| `make analyze` | courbes et tableau récapitulatif des expériences |
| `make clean` | supprime les caches |

| Script | Rôle |
|---|---|
| `scripts/run_evolution.py` | une évolution, enregistrée dans `results/` (`--scenario`, `--generations`, `--seed`...) |
| `scripts/replay.py` | rejoue le champion d'un fichier de résultats |
| `scripts/plot_run.py` | trace les courbes d'une évolution |
| `scripts/show_body.py` | affiche un corps au repos, pour vérifier un fichier YAML |
| `scripts/run_experiments.py` | lance une grille d'évolutions (créatures × scénarios × conditions × graines) |
| `scripts/analyze_experiments.py` | analyse la grille : `fitness.png`, `species.png`, `summary.csv` |
| `scripts/run_xor.py` | NEAT sur XOR |
| `scripts/search_sine.py` | recherche aléatoire d'un contrôleur sinus |

Les évolutions sont **déterministes** : avec la même graine (`--seed`), on obtient exactement le
même résultat, quel que soit le nombre de processus utilisés.

## Décrire une créature

Une créature est un fichier YAML dans `bodies/`. Elle apparaît automatiquement dans la démo, les
expériences et les tests. Unités : mètres, kilogrammes, degrés. Le sol est en y = 0, l'axe x vers
la droite, l'axe y vers le haut.

```yaml
name: Exemple
color: [200, 120, 80]     # couleur RVB

defaults:                 # valeurs utilisées quand un segment ou une articulation ne les donne pas
  segment: {width: 0.1, mass: 1.0, friction: 1.0}
  joint: {min_angle: -60, max_angle: 60, max_torque: 15, max_speed: 570, gain: 25}

segments:                 # des barres rigides, de "from" à "to"
  - {name: tete,  from: [0.0, 0.06], to: [0.5, 0.06]}
  - {name: corps, from: [0.5, 0.06], to: [1.0, 0.06], mass: 2.0}

joints:                   # des pivots motorisés entre deux segments
  - {between: [tete, corps], at: [0.5, 0.06]}
```

| Clé | Signification |
|---|---|
| `width`, `mass`, `friction` | épaisseur (m), masse (kg) et frottement d'un segment |
| `min_angle`, `max_angle` | butées de l'articulation, par rapport à la pose du fichier (°) |
| `max_torque` | couple maximal du moteur (N·m) |
| `max_speed` | vitesse maximale du moteur (°/s) |
| `gain` | raideur de l'asservissement : trop haut, l'articulation vibre ; trop bas, elle est molle |

## Le cerveau

À chaque pas de simulation, le réseau reçoit :

- l'angle de chaque articulation, ramené entre −1 et 1 entre ses butées,
- pour chaque segment, s'il touche le sol,
- les capteurs propres au scénario,
- une horloge, qui facilite les mouvements rythmiques.

Il renvoie une commande entre −1 et 1 par articulation, convertie en angle cible entre les butées.
Un asservissement proportionnel fait tourner le moteur vers cet angle.

## Les scénarios

| Scénario | Épreuves | Capteurs en plus | Fitness |
|---|---|---|---|
| Marche | 1 | aucun | distance parcourue vers la droite |
| Nage | 1 | aucun | distance parcourue vers la droite dans l'eau |
| Nourriture | 2 | direction et proximité de la boule | boules mangées, plus la progression vers la suivante (moyenne des deux épreuves) |
| Ballon | 2 | position et vitesse du ballon | distance parcourue par le ballon dans la bonne direction, plus un petit bonus d'approche. On mesure le **minimum** des deux épreuves, pour obliger à pousser des deux côtés |
| Rocher | 1 | avance sur le rocher, sa vitesse | temps de survie , l'épreuve s'arrête quand le rocher touche la créature |

Pour ajouter un scénario, il suffit d'écrire une classe dérivée de `Scenario` dans
`src/simsengine/sim/scenarios.py` et de l'ajouter au dictionnaire `SCENARIOS`. Les méthodes à
redéfinir au besoin :

| Méthode | Rôle |
|---|---|
| `variants()` | les épreuves sur lesquelles une créature est évaluée |
| `create_world()`, `setup(creature)` | construire le monde et y placer les objets |
| `apply_forces(creature)` | forces extérieures, à chaque sous-pas (eau, freinage...) |
| `extra_inputs(creature)` | capteurs propres au scénario (et `n_extra_inputs`) |
| `update(creature)` | ce qui se passe après chaque pas (manger, être rattrapé...) |
| `finished(creature)` | arrêter l'épreuve avant la fin |
| `fitness(creature, start_x)`, `combine(scores)` | note d'une épreuve, puis de l'ensemble |
| `hud_stats(creature, start_x)` | informations affichées pendant un replay |

Le dessin des objets d'un scénario (boules, ballon, rocher) se trouve dans
`src/simsengine/viz/renderer.py`.

## NEAT

L'implémentation suit l'article original :

- **innovations globales** : une même connexion ou un même noeud ajouté dans deux génomes reçoit le
  même numéro, ce qui permet d'aligner les génomes pour les croiser et les comparer ;
- **mutations** : perturbation des poids, ajout de connexion (sans créer de cycle), ajout de noeud
  (une connexion est coupée en deux) ;
- **spéciation** : distance `c1·E/N + c2·D/N + c3·W` (gènes en excès, disjoints, écart des poids),
  avec un **seuil adaptatif** qui vise 8 espèces ;
- **partage de fitness** entre les membres d'une espèce, **stagnation** (une espèce qui ne
  progresse plus pendant 15 générations disparaît, sauf les 2 meilleures), **élitisme** du
  champion de chaque espèce.

Tous les réglages sont dans `src/simsengine/neat/config.py` (population de 150 individus par
défaut). L'évaluation des individus est parallélisée sur tous les processeurs (`multiprocessing`).

## Architecture

```
bodies/                  les créatures
scripts/                 les points d'entrée en ligne de commande
src/simsengine/
├── neat/                l'algorithme NEAT
│   ├── config.py          tous les réglages
│   ├── genes.py           gènes de noeud et de connexion
│   ├── innovation.py      numéros d'innovation globaux
│   ├── genome.py          génome : mutations, distance, sauvegarde
│   ├── crossover.py       croisement de deux génomes
│   ├── network.py         génome -> réseau de neurones exécutable
│   ├── species.py         spéciation et seuil adaptatif
│   └── population.py      boucle d'évolution, reproduction, stagnation
├── sim/                 la simulation physique
│   ├── world.py           monde pymunk, sol, constantes de temps
│   ├── body.py            lecture des fichiers YAML
│   ├── creature.py        corps articulé, moteurs, capteurs
│   ├── controllers.py     contrôleurs NEAT et sinus
│   ├── scenarios.py       les 5 scénarios
│   └── simulation.py      une épreuve, pas à pas
├── viz/                 l'affichage et les courbes
│   ├── renderer.py        dessin du monde, des créatures et des objets
│   ├── demo.py            la démo interactive
│   └── ...                dessin bas niveau, décor, effets, courbes
├── evaluation.py        évaluation parallèle d'une population
├── experiment.py        enregistrement et lecture des résultats (JSON)
└── live.py              évolution en direct, dans un fil séparé (démo)
tests/                   tests (pytest)
```

## Choix de conception

- **2D plutôt que 3D** : la physique est plus rapide et plus stable, ce qui permet des milliers
  de simulations par évolution sur un ordinateur portable. La 2D suffit pour faire apparaître des
  comportements variés.
- **pymunk** : moteur physique éprouvé (Chipmunk), avec des articulations, des butées et des
  moteurs prêts à l'emploi.
- **Unités réelles** (mètres, kilogrammes) et **4 sous-pas physiques** par décision : sans eux, les
  articulations se déboîtaient sous l'effort.
- **Commandes normalisées** entre −1 et 1 : le même réseau fonctionne quelles que soient les
  butées des articulations.
- **Corps décrits en YAML** : ajouter une créature ne demande aucun code.

## Résultats

Les expériences (`make experiments`, `make speciation`, puis `make analyze`) donnent
principalement :

- **NEAT dépasse une recherche aléatoire de contrôleurs sinus** à budget égal (4 500 simulations)
  dans 21 combinaisons sur 25. L'écart est le plus grand quand il faut percevoir l'environnement
  (nourriture, ballon) ou garder l'équilibre (bipède). Le sinus fait aussi bien ou mieux quand la
  solution est un simple rythme : la nage du ver et du serpent, les bonds du sauteur.
- **La spéciation n'améliore pas les résultats** sur nos tâches de locomotion, alors qu'elle aide
  sur XOR. Les champions n'ont presque pas de noeuds cachés : les tâches se résolvent en réglant les
  poids, et la spéciation n'a pas d'innovation de structure à protéger.
- **NEAT exploite les failles du simulateur** quand il y en a (vibrations, articulations qui se
  déboîtent, ballon qui ne freine pas) : chacune a dû être corrigée.

## Limites

- en vue de profil, un serpent ne peut pas onduler latéralement comme un vrai serpent,
- les créatures trouvent la façon d'avancer la plus efficace pour la fitness, pas la plus
  naturelle : le bipède sautille au lieu de marcher,
- dans le scénario rocher, les créatures les plus rapides survivent jusqu'à la fin (10 s), et la
  fitness ne les départage plus,
- les expériences utilisent 3 à 5 graines : seules les différences importantes sont fiables.

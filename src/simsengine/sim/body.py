"""Description d'un corps de créature lue depuis le fichier YAML."""

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_COLOR = (91, 127, 214)


@dataclass
class SegmentSpec:
    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    width: float
    mass: float
    friction: float

    @property
    def length(self):
        return math.dist(self.start, self.end)

    @property
    def center(self):
        return ((self.start[0] + self.end[0]) / 2, (self.start[1] + self.end[1]) / 2)

    @property
    def angle(self):
        """Orientation du segment"""
        return math.atan2(self.end[1] - self.start[1], self.end[0] - self.start[0])


@dataclass
class JointSpec:
    a: str
    b: str
    anchor: tuple[float, float]
    min_angle: float
    max_angle: float
    max_torque: float
    max_speed: float
    gain: float


@dataclass
class BodySpec:
    name: str
    segments: list[SegmentSpec]
    joints: list[JointSpec]
    color: tuple[int, int, int] = DEFAULT_COLOR


def load_body(path):
    """Lit le fichier YAML et renvoie un BodySpec."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = data.get("defaults", {})
    seg_defaults = defaults.get("segment", {})
    joint_defaults = defaults.get("joint", {})

    def get(entry, key, fallback):
        """Valeur de l'entrée si elle est donnée, sinon valeur par défaut."""
        if key in entry:
            return entry[key]
        if key in fallback:
            return fallback[key]
        raise ValueError(f"{path} : '{key}' manquant dans {entry} (et pas de valeur par défaut)")

    segments = []
    for s in data["segments"]:
        segments.append(
            SegmentSpec(
                name=s["name"],
                start=tuple(s["from"]),
                end=tuple(s["to"]),
                width=get(s, "width", seg_defaults),
                mass=get(s, "mass", seg_defaults),
                friction=get(s, "friction", seg_defaults),
            )
        )

    names = [s.name for s in segments]
    if len(set(names)) != len(names):
        raise ValueError(f"{path} : deux segments portent le même nom")

    joints = []
    for j in data.get("joints", []):
        a, b = j["between"]
        for n in (a, b):
            if n not in names:
                raise ValueError(f"{path} : l'articulation {j} relie un segment inconnu « {n} »")
        min_angle = math.radians(get(j, "min_angle", joint_defaults))
        max_angle = math.radians(get(j, "max_angle", joint_defaults))
        if min_angle >= max_angle:
            raise ValueError(f"{path} : min_angle doit être inférieur à max_angle dans {j}")
        joints.append(
            JointSpec(
                a=a,
                b=b,
                anchor=tuple(j["at"]),
                min_angle=min_angle,
                max_angle=max_angle,
                max_torque=get(j, "max_torque", joint_defaults),
                max_speed=math.radians(get(j, "max_speed", joint_defaults)),
                gain=get(j, "gain", joint_defaults),
            )
        )

    color = tuple(data.get("color", DEFAULT_COLOR))
    if len(color) != 3:
        raise ValueError(f"{path} : 'color' doit être une liste [rouge, vert, bleu]")
    return BodySpec(name=path.stem, segments=segments, joints=joints, color=color)

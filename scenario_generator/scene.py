from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np

from .config import SceneConfig


@dataclass(frozen=True)
class Scene:
    """Ground-truth viewpoint graph and target samples.

    edges are undirected pairs. Their existence means that the direct flight
    connection has already been checked and is safe; obstacle geometry is not
    part of the online environment state.
    """

    viewpoint_positions: np.ndarray  # (N, 2)
    edges: tuple[tuple[int, int], ...]
    sample_positions: np.ndarray  # (M, 2)
    base_viewpoint: int = 0

    def __post_init__(self) -> None:
        vp = np.asarray(self.viewpoint_positions, dtype=np.float64)
        sp = np.asarray(self.sample_positions, dtype=np.float64)
        object.__setattr__(self, "viewpoint_positions", vp)
        object.__setattr__(self, "sample_positions", sp)
        if vp.ndim != 2 or vp.shape[1] != 2 or len(vp) < 2:
            raise ValueError("viewpoint_positions must have shape (N,2), N>=2")
        if sp.ndim != 2 or sp.shape[1] != 2 or len(sp) < 1:
            raise ValueError("sample_positions must have shape (M,2), M>=1")
        if not (0 <= self.base_viewpoint < len(vp)):
            raise ValueError("base_viewpoint is out of range")

        cleaned: set[tuple[int, int]] = set()
        for a, b in self.edges:
            a, b = int(a), int(b)
            if a == b or not (0 <= a < len(vp) and 0 <= b < len(vp)):
                raise ValueError(f"invalid edge {(a, b)}")
            cleaned.add((min(a, b), max(a, b)))
        object.__setattr__(self, "edges", tuple(sorted(cleaned)))

        graph = self.to_networkx()
        if not nx.is_connected(graph):
            raise ValueError("viewpoint graph must be connected")

    @property
    def num_viewpoints(self) -> int:
        return int(self.viewpoint_positions.shape[0])

    @property
    def num_samples(self) -> int:
        return int(self.sample_positions.shape[0])

    def edge_length(self, a: int, b: int) -> float:
        return float(np.linalg.norm(self.viewpoint_positions[a] - self.viewpoint_positions[b]))

    def adjacency_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.num_viewpoints, self.num_viewpoints), dtype=bool)
        for a, b in self.edges:
            matrix[a, b] = True
            matrix[b, a] = True
        return matrix

    def to_networkx(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_viewpoints))
        for a, b in self.edges:
            graph.add_edge(a, b, weight=self.edge_length(a, b))
        return graph


def _minimum_spanning_edges(points: np.ndarray) -> set[tuple[int, int]]:
    complete = nx.Graph()
    complete.add_nodes_from(range(len(points)))
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = float(np.linalg.norm(points[i] - points[j]))
            complete.add_edge(i, j, weight=d)
    tree = nx.minimum_spanning_tree(complete, weight="weight")
    return {(min(a, b), max(a, b)) for a, b in tree.edges}


def generate_random_scene(config: SceneConfig | None = None) -> Scene:
    """Create a connected random geometric viewpoint graph with random samples.

    This is a provisional generator, not a claim about the final paper setup.
    It contains no obstacles. Connectivity is guaranteed by an MST; additional
    geometric edges are added by radius and k-nearest rules.
    """

    cfg = config or SceneConfig()
    if cfg.num_viewpoints < 2 or cfg.num_samples < 1:
        raise ValueError("num_viewpoints>=2 and num_samples>=1 are required")
    rng = np.random.default_rng(cfg.seed)

    points = np.empty((cfg.num_viewpoints, 2), dtype=np.float64)
    points[0] = np.asarray(cfg.base_position, dtype=np.float64)
    points[1:, 0] = rng.uniform(0.0, cfg.area_width, size=cfg.num_viewpoints - 1)
    points[1:, 1] = rng.uniform(0.0, cfg.area_height, size=cfg.num_viewpoints - 1)

    edges = _minimum_spanning_edges(points)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    for i in range(cfg.num_viewpoints):
        for j in range(i + 1, cfg.num_viewpoints):
            if distances[i, j] <= cfg.connection_radius:
                edges.add((i, j))
        nearest = np.argsort(distances[i])[1 : 1 + max(0, cfg.extra_k_nearest)]
        for j in nearest:
            edges.add((min(i, int(j)), max(i, int(j))))

    min_base_degree = max(0, int(cfg.min_base_degree))
    if min_base_degree >= cfg.num_viewpoints:
        raise ValueError("min_base_degree must be smaller than num_viewpoints")
    base = 0
    base_neighbors = {b if a == base else a for a, b in edges if a == base or b == base}
    if len(base_neighbors) < min_base_degree:
        for target in np.argsort(distances[base])[1:]:
            target = int(target)
            if target == base or target in base_neighbors:
                continue
            edges.add((min(base, target), max(base, target)))
            base_neighbors.add(target)
            if len(base_neighbors) >= min_base_degree:
                break

    samples = np.column_stack(
        [
            rng.uniform(0.0, cfg.area_width, size=cfg.num_samples),
            rng.uniform(0.0, cfg.area_height, size=cfg.num_samples),
        ]
    )
    return Scene(points, tuple(sorted(edges)), samples, base_viewpoint=0)


def save_scene_json(
    scene: Scene,
    path: str | Path,
    *,
    indent: int | None = 2,
) -> None:
    data = {
        "viewpoint_positions": scene.viewpoint_positions.tolist(),
        "edges": [list(edge) for edge in scene.edges],
        "sample_positions": scene.sample_positions.tolist(),
        "base_viewpoint": scene.base_viewpoint,
    }
    separators = (",", ":") if indent is None else None
    Path(path).write_text(
        json.dumps(data, indent=indent, separators=separators) + "\n",
        encoding="utf-8",
        newline="\r\n",
    )


def load_scene_json(path: str | Path) -> Scene:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Scene(
        viewpoint_positions=np.asarray(data["viewpoint_positions"], dtype=np.float64),
        edges=tuple(tuple(map(int, edge)) for edge in data["edges"]),
        sample_positions=np.asarray(data["sample_positions"], dtype=np.float64),
        base_viewpoint=int(data.get("base_viewpoint", 0)),
    )

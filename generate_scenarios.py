"""Generate deterministic, disjoint publication scene splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np


from scenario_generator.config import SceneConfig
from scenario_generator.scene import Scene, generate_random_scene, save_scene_json


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
BASE_CONFIG = SceneConfig(num_viewpoints=32, num_samples=80)
DIFFICULTY_PROFILES: dict[str, dict[str, Any]] = {
    "easy": {
        "area_width": 18.0,
        "area_height": 18.0,
        "connection_radius": 6.0,
        "extra_k_nearest": 3,
        "min_base_degree": 4,
    },
    "medium": {
        "area_width": 20.0,
        "area_height": 20.0,
        "connection_radius": 5.0,
        "extra_k_nearest": 2,
        "min_base_degree": 3,
    },
    "hard": {
        "area_width": 22.0,
        "area_height": 22.0,
        "connection_radius": 4.25,
        "extra_k_nearest": 1,
        # Three UAVs require three distinct legal initial departures.
        "min_base_degree": 3,
    },
}
SPLIT_OFFSETS = {"train": 0, "validation": 100_000, "test": 200_000}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene_stats(scene: Scene) -> dict[str, Any]:
    graph = scene.to_networkx()
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    hop_lengths = [
        length
        for source, lengths in nx.all_pairs_shortest_path_length(graph)
        for target, length in lengths.items()
        if source < target
    ]
    weighted_from_base = nx.single_source_dijkstra_path_length(
        graph, scene.base_viewpoint, weight="weight"
    )
    return {
        "num_viewpoints": scene.num_viewpoints,
        "num_samples": scene.num_samples,
        "num_edges": graph.number_of_edges(),
        "graph_density": float(nx.density(graph)),
        "mean_degree": float(degrees.mean()),
        "min_degree": int(degrees.min()),
        "max_degree": int(degrees.max()),
        "diameter_hops": int(nx.diameter(graph)),
        "mean_shortest_path_hops": float(np.mean(hop_lengths)),
        "max_base_distance": float(max(weighted_from_base.values())),
        "base_degree": int(graph.degree(scene.base_viewpoint)),
    }


def generate_split(
    output_root: Path,
    split: str,
    count: int,
    master_seed: int,
    max_team_size: int = 3,
) -> list[dict[str, Any]]:
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=False)
    profile_names = tuple(DIFFICULTY_PROFILES)
    rows: list[dict[str, Any]] = []
    for local_id in range(count):
        difficulty = profile_names[local_id % len(profile_names)]
        generator_seed = (
            int(master_seed) + SPLIT_OFFSETS[split] + local_id
        )
        profile = dict(DIFFICULTY_PROFILES[difficulty])
        profile["min_base_degree"] = max(profile["min_base_degree"], max_team_size)
        config = replace(
            BASE_CONFIG,
            seed=generator_seed,
            **profile,
        )
        scene = generate_random_scene(config)
        path = split_dir / f"scene_seed_{local_id}.json"
        # Publication sets contain many scenes; compact JSON keeps the tracked
        # artefact small without changing any numeric value.
        save_scene_json(scene, path, indent=None)
        rows.append(
            {
                "scene_id": local_id,
                "generator_seed": generator_seed,
                "difficulty": difficulty,
                "relative_path": path.relative_to(output_root).as_posix(),
                "sha256": file_sha256(path),
                "generator_config": asdict(config),
                "statistics": scene_stats(scene),
            }
        )
    return rows


def generate_dataset(
    output_root: Path,
    train_count: int,
    validation_count: int,
    test_count: int,
    master_seed: int,
    max_team_size: int = 3,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing scene set: {output_root}"
        )
    counts = {
        "train": int(train_count),
        "validation": int(validation_count),
        "test": int(test_count),
    }
    if any(value <= 0 for value in counts.values()):
        raise ValueError("all split counts must be positive")
    if not 2 <= max_team_size < BASE_CONFIG.num_viewpoints:
        raise ValueError("max_team_size must be between 2 and num_viewpoints - 1")
    if any(value > 100_000 for value in counts.values()):
        raise ValueError("split counts exceed disjoint generator-seed ranges")
    output_root.mkdir(parents=True)
    splits = {
        split: generate_split(output_root, split, count, master_seed, max_team_size)
        for split, count in counts.items()
    }
    seeds = [
        row["generator_seed"]
        for rows in splits.values()
        for row in rows
    ]
    hashes = [row["sha256"] for rows in splits.values() for row in rows]
    if len(seeds) != len(set(seeds)) or len(hashes) != len(set(hashes)):
        raise RuntimeError("generated split contains duplicate seeds or scenes")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "master_seed": int(master_seed),
        "fixed_dimensions": {"num_viewpoints": 32, "num_samples": 80},
        "max_team_size": max_team_size,
        "difficulty_profiles": {
            key: {**value, "min_base_degree": max(value["min_base_degree"], max_team_size)}
            for key, value in DIFFICULTY_PROFILES.items()
        },
        "policy": {
            "train_and_validation_may_be_used_for_development": True,
            "test_is_frozen_and_must_not_be_used_for_tuning": True,
        },
        "splits": splits,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\r\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "v2_team4.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir
    manifest = generate_dataset(output, **config)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "counts": {
                    key: len(value) for key, value in manifest["splits"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

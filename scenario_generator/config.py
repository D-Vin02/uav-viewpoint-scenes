from dataclasses import dataclass


@dataclass
class SceneConfig:
    """Provisional scene-generation settings.

    These defaults are deliberately isolated because the source specification
    leaves viewpoint/sample generation and edge-generation rules open.
    No obstacle grid is created. An edge itself means that the straight flight
    connection has already been certified as safe.
    """

    num_viewpoints: int = 36
    num_samples: int = 90
    area_width: float = 20.0
    area_height: float = 20.0
    connection_radius: float = 5.0
    extra_k_nearest: int = 2
    min_base_degree: int = 3
    base_position: tuple[float, float] = (1.0, 1.0)
    seed: int = 7

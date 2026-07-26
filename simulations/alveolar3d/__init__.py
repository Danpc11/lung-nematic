"""Genuinely three-dimensional alveolar fibrosis prototype."""

from .model import (
    ABERRANT,
    AT1,
    AT2,
    COLLAPSED,
    EMPTY,
    INDURATED,
    KRT8,
    OPEN,
    HUMAN_CHRONIC_RATE_SCALE,
    HUMAN_CHRONIC_REFERENCE_DAYS,
    Alveolar3DConfig,
    Alveolar3DSimulation,
    accelerated_3d_demo_config,
    compact_alveolar_centres,
    human_chronic_3d_config,
)
from .render import (
    Alveolar3DRenderConfig,
    draw_3d_frame,
    inspiration_fraction,
    run_and_record_3d,
)

__all__ = [
    "EMPTY",
    "AT1",
    "AT2",
    "KRT8",
    "ABERRANT",
    "OPEN",
    "COLLAPSED",
    "INDURATED",
    "HUMAN_CHRONIC_RATE_SCALE",
    "HUMAN_CHRONIC_REFERENCE_DAYS",
    "Alveolar3DConfig",
    "Alveolar3DSimulation",
    "accelerated_3d_demo_config",
    "human_chronic_3d_config",
    "compact_alveolar_centres",
    "Alveolar3DRenderConfig",
    "inspiration_fraction",
    "draw_3d_frame",
    "run_and_record_3d",
]

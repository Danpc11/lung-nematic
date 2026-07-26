"""Genuinely three-dimensional alveolar fibrosis prototype."""

from .model import (
    ABERRANT,
    AT1,
    AT2,
    COLLAPSED,
    EMPTY,
    HUMAN_CHRONIC_RATE_SCALE,
    HUMAN_CHRONIC_REFERENCE_DAYS,
    INDURATED,
    KRT8,
    OPEN,
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
    "ABERRANT",
    "AT1",
    "AT2",
    "COLLAPSED",
    "EMPTY",
    "HUMAN_CHRONIC_RATE_SCALE",
    "HUMAN_CHRONIC_REFERENCE_DAYS",
    "INDURATED",
    "KRT8",
    "OPEN",
    "Alveolar3DConfig",
    "Alveolar3DRenderConfig",
    "Alveolar3DSimulation",
    "accelerated_3d_demo_config",
    "compact_alveolar_centres",
    "draw_3d_frame",
    "human_chronic_3d_config",
    "inspiration_fraction",
    "run_and_record_3d",
]

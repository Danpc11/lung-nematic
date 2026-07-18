"""Alveolar epithelium stage of the pulmonary fibrosis model."""

from .geometry import AlveolarGeometry, Alveolus, Septum
from .model import (
    ABERRANT, AT1, AT2, COLLAPSED, EMPTY, INDURATED, KRT8, OPEN,
    STATE_NAMES, AlveolarConfig, AlveolarSimulation,
)
from .render import draw_frame, run_and_record

__all__ = [
    "AlveolarGeometry", "Alveolus", "Septum",
    "AlveolarConfig", "AlveolarSimulation",
    "AT1", "AT2", "KRT8", "ABERRANT", "EMPTY",
    "OPEN", "COLLAPSED", "INDURATED", "STATE_NAMES",
    "run_and_record", "draw_frame",
]
__version__ = "0.1.0"

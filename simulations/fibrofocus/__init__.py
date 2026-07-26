"""Active-nematic model of fibroblastic focus formation in pulmonary fibrosis."""

from .bistability import (
    critical_value,
    fixed_points,
    integrate_lesion,
    scan_two_parameters,
    stiffness_velocity,
)
from .model import FocusConfig, FocusSimulation
from .render import detect_defects, draw_frame, run_and_record

__all__ = [
    "FocusConfig",
    "FocusSimulation",
    "critical_value",
    "detect_defects",
    "draw_frame",
    "fixed_points",
    "integrate_lesion",
    "run_and_record",
    "scan_two_parameters",
    "stiffness_velocity",
]
__version__ = "0.1.0"

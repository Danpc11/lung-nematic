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
    "FocusConfig", "FocusSimulation",
    "fixed_points", "stiffness_velocity", "integrate_lesion",
    "scan_two_parameters", "critical_value",
    "run_and_record", "draw_frame", "detect_defects",
]
__version__ = "0.1.0"

"""lungtwin - identifiability tooling for a low-dimensional IPF digital twin.

Deliberately not an estimator. The package answers what *can* be estimated from
a given study design before any estimator exists, because a fit to a
structurally unidentifiable model does not fail loudly: it returns whatever
point the optimiser drifted to, and it looks like a result.
"""

from .design import DEFAULT_NOISE_SD, VisitSchedule, routine_followup
from .identifiability import (
    IdentifiabilityReport,
    analyze,
    monte_carlo_check,
    profile_likelihood,
    sensitivity_matrix,
    visits_required,
)
from .model import Channel, Parameters, ReserveMode, observe, simulate
from .report import render

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_NOISE_SD",
    "Channel",
    "IdentifiabilityReport",
    "Parameters",
    "ReserveMode",
    "VisitSchedule",
    "analyze",
    "monte_carlo_check",
    "observe",
    "profile_likelihood",
    "render",
    "routine_followup",
    "sensitivity_matrix",
    "simulate",
    "visits_required",
]

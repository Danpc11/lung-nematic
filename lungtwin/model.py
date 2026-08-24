"""The reparameterized two-state IPF progression model.

Scale fixing
------------
The latent burden ``B`` has no natural units, so the original specification
(``FVC = FVC0 - c_F*B``, ``B(0) = B0``) could only ever determine the products
``c_F*r_i`` and ``c_F*beta``, never their factors, and ``B0`` traded off against
``FVC0`` and ``DLCO0``. Two conventions remove all of that without changing the
model's predictions:

    c_F == 1     ->  B is measured in FVC percentage points
    B(0) == 0    ->  FVC0 is literally the baseline FVC % predicted

What survives is a single coupling ratio ``kappa = c_D / c_F``, which is
identifiable and clinically meaningful: it is the diffusion-per-volume slope,
and it is where combined pulmonary fibrosis and emphysema (CPFE) patients leave
the model, since they lose DLCO while preserving volumes.

The reserve state
-----------------
In the original specification ``R`` appeared in neither the burden equation nor
any observation equation, so its sensitivity was identically zero: ``R0`` and
``alpha`` were structurally unidentifiable, and deleting the whole state changed
no prediction. ``ReserveMode`` makes the three possible resolutions explicit:

``NONE``
    Drop ``R`` entirely. Honest baseline. Note that this makes ``B`` linear in
    time, so FVC declines linearly and the model reduces algebraically to
    per-patient linear regression; any gain over that baseline comes from
    partial pooling and from the joint use of DLCO, not from mechanism.
``OBSERVED``
    ``R`` gets its own measurement channel (resting SpO2, or 6MWT
    desaturation). Reserve becomes identifiable through data rather than
    through structure.
``FEEDBACK``
    ``R`` modulates the burden rate: less reserve means more mechanical load on
    the remaining units, so progression accelerates. This is the mechanosensitive
    hypothesis, and it is the only mode in which the mechanistic layer does work
    a hierarchical linear model could not.

``FEEDBACK`` also needs a scale convention, so ``R(0) == 1`` (dimensionless
reserve fraction) and ``alpha`` carries the units.

Treatment exposure
------------------
``B(t) = int_0^t (r_i - beta*T(s)) f(R(s)) ds``. The integral matters: a naive
discretisation that evaluates treatment only at visit times silently deletes the
pre-treatment interval, and with it the only window in which ``beta`` is
separable from ``r_i``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, fields, replace
from enum import Enum

import numpy as np
from scipy.integrate import solve_ivp


class ReserveMode(str, Enum):
    NONE = "none"
    OBSERVED = "observed"
    FEEDBACK = "feedback"


class Channel(str, Enum):
    FVC = "fvc_pct"
    DLCO = "dlco_pct"
    SPO2 = "spo2"


@dataclass(frozen=True)
class Parameters:
    """Model parameters. Defaults are population anchors, not fitted values.

    ``r_i`` and ``kappa`` are set from the PROFILE incident cohort's 12-month
    estimates (-5.28 ppFVC, -3.35 ppDLCO), giving kappa ~ 0.63. Note the
    direction: in percentage-point terms DLCO falls *more slowly* than FVC,
    which is the opposite of the usual intuition. ``beta`` corresponds to the
    ~50% reduction in decline seen in the antifibrotic trials.
    """

    fvc0: float = 73.0        # % predicted at baseline
    dlco0: float = 58.0       # % predicted at baseline
    r_i: float = 5.28         # untreated burden accrual, FVC points / year
    kappa: float = 0.63       # DLCO points per FVC point
    beta: float = 2.64        # absolute reduction in accrual while treated
    alpha: float = 0.010      # reserve lost per FVC point of burden per year
    gamma: float = 0.0        # feedback exponent; 0 == no feedback
    spo2_0: float = 95.0      # % at baseline
    lam: float = 12.0         # SpO2 points lost per unit of reserve lost

    def names(self) -> list[str]:
        return [item.name for item in fields(self)]

    def as_array(self, names: list[str]) -> np.ndarray:
        return np.array([getattr(self, name) for name in names], dtype=float)

    def with_values(self, names: list[str], values) -> Parameters:
        return replace(self, **dict(zip(names, map(float, values))))


def _reserve_factor(reserve: float, gamma: float) -> float:
    """f(R): burden accrual multiplier. Decreasing in reserve when gamma > 0."""
    if gamma == 0.0:
        return 1.0
    return float(max(reserve, 1e-6) ** (-gamma))


def simulate(
    times: np.ndarray,
    params: Parameters,
    treatment_start: float | None,
    mode: ReserveMode = ReserveMode.NONE,
) -> dict[str, np.ndarray]:
    """Integrate the model and return the latent states at ``times``.

    ``treatment_start`` is in years from baseline; ``None`` means never treated.
    Integration is done with an explicit breakpoint at the treatment start so
    the discontinuity in T(t) is not smeared by adaptive stepping.
    """
    times = np.asarray(times, dtype=float)
    if times[0] != 0.0:
        raise ValueError("times must start at 0 (baseline defines B(0) = 0).")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing.")

    switch = np.inf if treatment_start is None else float(treatment_start)

    def rhs(t, y, treated):
        burden, reserve = y
        rate = params.r_i - params.beta * treated
        if mode is ReserveMode.FEEDBACK:
            rate *= _reserve_factor(reserve, params.gamma)
        return [rate, -params.alpha * burden]

    # T(t) is a step function. Adaptive steppers do not see the discontinuity
    # and will smear it, which shrinks the pre-treatment window - the only
    # window in which beta separates from r_i. Integrate each treatment regime
    # as its own initial value problem instead, with the switch as a hard
    # boundary, and evaluate the requested times within their own segment.
    edges = [0.0]
    if 0.0 < switch < times[-1]:
        edges.append(switch)
    edges.append(float(times[-1]) if times[-1] > 0.0 else 1.0)

    burden = np.full(times.shape, np.nan)
    reserve = np.full(times.shape, np.nan)
    burden[times == 0.0] = 0.0
    reserve[times == 0.0] = 1.0

    state = np.array([0.0, 1.0])
    for start, stop in itertools.pairwise(edges):
        if stop <= start:
            continue
        treated = 1.0 if start >= switch else 0.0
        # Half-open on the left, closed on the right, uniformly. A visit landing
        # exactly on the switch therefore belongs to the pre-treatment segment;
        # B is continuous there so the value is the same either way, but the
        # assignment must be unambiguous or the point falls through both
        # segments and is never evaluated.
        inside = (times > start) & (times <= stop)
        # Always evaluate the segment endpoint too, so the carried state comes
        # from this same solve rather than a second integration.
        wanted = np.unique(np.concatenate([times[inside], [stop]]))
        solution = solve_ivp(
            rhs,
            (start, stop),
            y0=state,
            t_eval=wanted,
            args=(treated,),
            rtol=1e-11,
            atol=1e-13,
        )
        if not solution.success:  # pragma: no cover - solver failure is a bug
            raise RuntimeError(f"integration failed: {solution.message}")
        for column, time in enumerate(wanted):
            hit = inside & (times == time)
            if hit.any():
                burden[hit] = solution.y[0][column]
                reserve[hit] = solution.y[1][column]
        state = solution.y[:, -1].copy()

    if np.isnan(burden).any():  # pragma: no cover - defensive
        raise RuntimeError("some requested times were not evaluated")

    return {"burden": burden, "reserve": reserve}


def observe(
    times: np.ndarray,
    params: Parameters,
    treatment_start: float | None,
    mode: ReserveMode = ReserveMode.NONE,
    channels: tuple[Channel, ...] = (Channel.FVC, Channel.DLCO),
) -> dict[str, np.ndarray]:
    """Noise-free observation channels at ``times``."""
    states = simulate(times, params, treatment_start, mode)
    burden, reserve = states["burden"], states["reserve"]

    out: dict[str, np.ndarray] = {}
    for channel in channels:
        if channel is Channel.FVC:
            out[channel.value] = params.fvc0 - burden
        elif channel is Channel.DLCO:
            out[channel.value] = params.dlco0 - params.kappa * burden
        elif channel is Channel.SPO2:
            if mode is ReserveMode.NONE:
                raise ValueError(
                    "SpO2 observes the reserve state, which ReserveMode.NONE "
                    "removes from the model."
                )
            out[channel.value] = params.spo2_0 - params.lam * (1.0 - reserve)
    return out

"""Structural and practical identifiability for the IPF twin.

Two questions, and they are not the same one.

*Structural* identifiability asks whether the parameters could be recovered from
noise-free data of the given design. It is a property of the model and the
experiment, not of the data: if the sensitivity matrix is rank deficient, some
combination of parameters leaves every observation unchanged, and no amount of
data, no prior, and no optimiser will recover it. Fitting anyway does not fail
loudly - the optimiser returns whatever point the initialisation drifted to, and
it looks like an estimate.

*Practical* identifiability asks how precisely the parameters can be recovered
given a realistic number of visits and known measurement noise. A model can be
structurally identifiable and still useless if the standard error on ``r_i``
exceeds the effect you want to detect. This is answered by the Cramer-Rao lower
bound, and it is the question that decides how many visits a study needs.

Run both before writing an estimator. The order matters: a practical bound
computed on a structurally unidentifiable model is meaningless, because the
information matrix is singular.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .design import VisitSchedule
from .model import Parameters, ReserveMode, observe


@dataclass
class IdentifiabilityReport:
    free_names: list[str]
    singular_values: np.ndarray
    rank: int
    null_directions: list[dict[str, float]]
    n_observations: int
    standard_errors: dict[str, float] | None
    correlations: np.ndarray | None
    design: str

    @property
    def is_structurally_identifiable(self) -> bool:
        return self.rank == len(self.free_names)

    @property
    def condition_number(self) -> float:
        positive = self.singular_values[self.singular_values > 0]
        if positive.size == 0:
            return float("inf")
        return float(positive[0] / positive[-1])


def sensitivity_matrix(
    params: Parameters,
    free_names: list[str],
    schedule: VisitSchedule,
    mode: ReserveMode = ReserveMode.NONE,
    *,
    scale_by_noise: bool = True,
    relative_step: float = 1e-6,
) -> np.ndarray:
    """Central-difference sensitivities, optionally scaled to sqrt(Fisher info).

    Rows are observations (stacked across channels, missing entries dropped),
    columns are free parameters. With ``scale_by_noise`` each row is divided by
    that observation's measurement standard deviation, which both makes channels
    in different units comparable and makes ``S.T @ S`` the Fisher information
    matrix for Gaussian noise.
    """
    _validate_free_names(params, free_names, mode)

    def predict(vector: np.ndarray) -> np.ndarray:
        trial = params.with_values(free_names, vector)
        return schedule.stack(
            observe(
                schedule.times,
                trial,
                schedule.treatment_start,
                mode,
                schedule.channels,
            )
        )

    base = params.as_array(free_names)
    columns = []
    for index in range(base.size):
        step = relative_step * max(1.0, abs(base[index]))
        up, down = base.copy(), base.copy()
        up[index] += step
        down[index] -= step
        columns.append((predict(up) - predict(down)) / (2.0 * step))
    matrix = np.column_stack(columns)

    if scale_by_noise:
        matrix = matrix / schedule.noise_vector()[:, None]
    return matrix


def analyze(
    params: Parameters,
    free_names: list[str],
    schedule: VisitSchedule,
    mode: ReserveMode = ReserveMode.NONE,
    *,
    rank_tol: float = 1e-8,
) -> IdentifiabilityReport:
    """Full structural + practical identifiability analysis for one design."""
    matrix = sensitivity_matrix(params, free_names, schedule, mode)
    _, singular, right = np.linalg.svd(matrix, full_matrices=True)

    # Pad the singular value spectrum when there are fewer observations than
    # parameters, so the rank deficiency is visible rather than implied.
    spectrum = np.zeros(len(free_names))
    spectrum[: singular.size] = singular

    largest = spectrum[0] if spectrum.size and spectrum[0] > 0 else 1.0
    rank = int((spectrum > rank_tol * largest).sum())

    null_directions = []
    for index in range(rank, len(free_names)):
        vector = right[index]
        vector = vector / np.abs(vector).max()
        null_directions.append(
            {
                name: float(np.round(value, 4))
                for name, value in zip(free_names, vector)
                if abs(value) > 1e-6
            }
        )

    standard_errors = None
    correlations = None
    if rank == len(free_names):
        information = matrix.T @ matrix
        covariance = np.linalg.inv(information)
        errors = np.sqrt(np.diag(covariance))
        standard_errors = dict(zip(free_names, map(float, errors)))
        correlations = covariance / np.outer(errors, errors)

    return IdentifiabilityReport(
        free_names=list(free_names),
        singular_values=spectrum,
        rank=rank,
        null_directions=null_directions,
        n_observations=schedule.n_observations,
        standard_errors=standard_errors,
        correlations=correlations,
        design=schedule.describe(),
    )


def visits_required(
    params: Parameters,
    free_names: list[str],
    target: dict[str, float],
    *,
    mode: ReserveMode = ReserveMode.NONE,
    interval_months: float = 6.0,
    treatment_start: float | None = None,
    channels=None,
    max_visits: int = 40,
) -> dict[str, int | None]:
    """Smallest visit count whose CRLB meets each target standard error.

    This is the design question stated in the units a study protocol uses. A
    target of ``{"r_i": 1.0}`` asks: how long must a patient be followed before
    their personal decline rate is pinned to within one FVC point per year?
    Returns ``None`` for any parameter whose target is unreachable within
    ``max_visits``, which is itself a useful answer.
    """
    from .design import routine_followup

    if channels is None:
        from .model import Channel

        channels = (Channel.FVC, Channel.DLCO)

    found: dict[str, int | None] = {name: None for name in target}
    for n_visits in range(2, max_visits + 1):
        schedule = routine_followup(
            n_visits=n_visits,
            interval_months=interval_months,
            treatment_start=treatment_start,
            channels=channels,
        )
        report = analyze(params, free_names, schedule, mode)
        if report.standard_errors is None:
            continue
        for name, threshold in target.items():
            if found[name] is None and report.standard_errors[name] <= threshold:
                found[name] = n_visits
        if all(value is not None for value in found.values()):
            break
    return found


def monte_carlo_check(
    params: Parameters,
    free_names: list[str],
    schedule: VisitSchedule,
    mode: ReserveMode = ReserveMode.NONE,
    *,
    n_replicates: int = 400,
    seed: int = 0,
    collapse_tolerance: float = 2.0,
) -> dict:
    """Refit synthetic replicates and compare empirical spread to the CRLB.

    The Cramer-Rao bound is a *local* quantity: it linearises the model at the
    true parameters and is blind to failure modes that live elsewhere in
    parameter space. This model has one. ``kappa`` multiplies the burden, so it
    is only identifiable when burden actually accrues; in replicates where the
    fitted trajectory happens to be nearly flat, ``kappa`` is unconstrained and
    runs off to arbitrary values with no likelihood penalty. The CRLB reports a
    perfectly reasonable standard error while the empirical standard deviation
    is three orders of magnitude larger.

    That is not a bug in the bound, and it is not noise - it is the model
    telling you that the DLCO-to-FVC coupling cannot be estimated in a patient
    who is not declining. Clinically that is exactly right, and it means
    ``kappa`` must be partially pooled across patients rather than estimated
    per patient.

    Returns both the classical standard deviation and a median-absolute-
    deviation estimate. When they disagree by a large factor, the difference is
    the diagnostic, and ``collapse_fraction`` names how often it happened.
    """
    from scipy.optimize import least_squares

    report = analyze(params, free_names, schedule, mode)
    sd = schedule.noise_vector()
    truth = schedule.stack(
        observe(schedule.times, params, schedule.treatment_start, mode,
                schedule.channels)
    )
    rng = np.random.default_rng(seed)
    start = params.as_array(free_names)

    estimates = np.empty((n_replicates, len(free_names)))
    collapsed = np.zeros(n_replicates, dtype=bool)
    horizon = schedule.times[-1]

    for replicate in range(n_replicates):
        data = truth + rng.normal(0.0, sd)

        def residual(vector, data=data):
            trial = params.with_values(free_names, vector)
            predicted = schedule.stack(
                observe(schedule.times, trial, schedule.treatment_start, mode,
                        schedule.channels)
            )
            return (data - predicted) / sd

        result = least_squares(residual, x0=start, method="lm",
                               xtol=1e-12, ftol=1e-12)
        estimates[replicate] = result.x
        fitted = params.with_values(free_names, result.x)
        burden_end = simulate_burden_end(fitted, schedule, mode, horizon)
        collapsed[replicate] = abs(burden_end) < collapse_tolerance

    median = np.median(estimates, axis=0)
    mad_sd = 1.4826 * np.median(np.abs(estimates - median), axis=0)

    return {
        "crlb": report.standard_errors,
        "empirical_sd": dict(zip(free_names,
                                 map(float, estimates.std(axis=0, ddof=1)))),
        "robust_sd": dict(zip(free_names, map(float, mad_sd))),
        "median_estimate": dict(zip(free_names, map(float, median))),
        "collapse_fraction": float(collapsed.mean()),
        "n_replicates": n_replicates,
    }


def simulate_burden_end(
    params: Parameters,
    schedule: VisitSchedule,
    mode: ReserveMode,
    horizon: float,
) -> float:
    """Burden accrued by the end of follow-up, used as a collapse detector."""
    from .model import simulate

    times = np.array([0.0, horizon]) if horizon > 0 else np.array([0.0])
    return float(
        simulate(times, params, schedule.treatment_start, mode)["burden"][-1]
    )


def profile_likelihood(
    params: Parameters,
    free_names: list[str],
    schedule: VisitSchedule,
    target: str,
    grid: np.ndarray,
    mode: ReserveMode = ReserveMode.NONE,
    *,
    seed: int = 0,
    noisy: bool = False,
) -> np.ndarray:
    """Profile the negative log-likelihood over ``target``, refitting the rest.

    A flat profile is the empirical signature of non-identifiability, and unlike
    the SVD it also catches the *practically* flat case where the model is
    technically full rank but the likelihood is indistinguishable from flat over
    the plausible range. Run with ``noisy=False`` first: on noise-free synthetic
    data a truly identifiable parameter must show a sharp minimum at the truth,
    so a flat noise-free profile is unambiguously a structural problem rather
    than a data problem.
    """
    from scipy.optimize import minimize

    _validate_free_names(params, free_names, mode)
    if target not in free_names:
        raise ValueError(f"{target!r} is not among the free parameters")

    truth = schedule.stack(
        observe(schedule.times, params, schedule.treatment_start, mode,
                schedule.channels)
    )
    sd = schedule.noise_vector()
    if noisy:
        rng = np.random.default_rng(seed)
        data = truth + rng.normal(0.0, sd)
    else:
        data = truth

    nuisance = [name for name in free_names if name != target]
    index = free_names.index(target)

    def nll(vector: np.ndarray) -> float:
        trial = params.with_values(free_names, vector)
        predicted = schedule.stack(
            observe(schedule.times, trial, schedule.treatment_start, mode,
                    schedule.channels)
        )
        return float(0.5 * np.sum(((data - predicted) / sd) ** 2))

    profile = np.empty(grid.size)
    start = params.as_array(free_names)
    for position, value in enumerate(grid):
        if not nuisance:
            vector = start.copy()
            vector[index] = value
            profile[position] = nll(vector)
            continue

        def partial(free_vector, value=value):
            vector = start.copy()
            vector[index] = value
            for name, entry in zip(nuisance, free_vector):
                vector[free_names.index(name)] = entry
            return nll(vector)

        result = minimize(
            partial,
            x0=params.as_array(nuisance),
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000,
                     "maxfev": 20000},
        )
        profile[position] = float(result.fun)
    return profile


def _validate_free_names(
    params: Parameters, free_names, mode: ReserveMode
) -> None:
    known = set(params.names())
    unknown = [name for name in free_names if name not in known]
    if unknown:
        raise ValueError(f"unknown parameter name(s): {unknown}")
    if len(set(free_names)) != len(free_names):
        raise ValueError("free_names contains duplicates")

    # Guard the scale conventions. Freeing a parameter that the reparameterization
    # fixed silently reintroduces the non-identifiability it was meant to remove,
    # and the failure would look like a plausible but arbitrary fit.
    if mode is ReserveMode.NONE:
        for name in ("alpha", "gamma", "spo2_0", "lam"):
            if name in free_names:
                raise ValueError(
                    f"{name!r} cannot be free under ReserveMode.NONE: the "
                    "reserve state is absent, so its sensitivity is "
                    "identically zero."
                )
    if mode is ReserveMode.OBSERVED and "gamma" in free_names:
        raise ValueError(
            "'gamma' is the reserve feedback exponent and has no effect under "
            "ReserveMode.OBSERVED; use ReserveMode.FEEDBACK."
        )
    if mode is ReserveMode.FEEDBACK:
        for name in ("spo2_0", "lam"):
            if name in free_names:
                raise ValueError(
                    f"{name!r} parameterizes the SpO2 channel, which "
                    "ReserveMode.FEEDBACK does not observe."
                )

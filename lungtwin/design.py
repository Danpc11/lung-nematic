"""Visit schedules, channel availability, and measurement noise.

Measurement noise is *pinned*, not estimated. With three or four visits,
measurement error in FVC and process noise in the burden state both produce
scatter around the trajectory and are not jointly identifiable. Spirometry and
DLCO repeatability are well characterised externally, so the defensible move is
to fix the measurement standard deviations from that literature and let the
process noise carry whatever is left. Every default below is a stand-in that the
user should replace with their own laboratory's repeatability; they are wired
through explicitly so the assumption is visible rather than buried.

Noise also sets the *scale* of the sensitivity analysis. A raw sensitivity matrix
mixes channels measured in different units and with different reliability; a
parameter that moves DLCO by one point is worth far less than one that moves FVC
by one point, because DLCO is noisier. Dividing each row by its standard
deviation makes the matrix the square root of the Fisher information, so its
singular values are comparable across channels and its inverse gives real
standard errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Channel

# Within-subject repeatability standard deviations, in percentage points of
# predicted (SpO2 in absolute percent). Replace with local values.
DEFAULT_NOISE_SD: dict[str, float] = {
    Channel.FVC.value: 3.0,
    Channel.DLCO.value: 5.0,
    Channel.SPO2.value: 2.0,
}


@dataclass
class VisitSchedule:
    """When the patient is seen, what is measured, and when treatment starts.

    Parameters
    ----------
    times:
        Visit times in years from baseline. Must start at 0.
    channels:
        Which observation channels exist at all for this patient.
    treatment_start:
        Years from baseline at which an antifibrotic is started; ``None`` for
        never treated, ``0.0`` for already treated at baseline. This single
        field decides whether ``beta`` is estimable: with a constant treatment
        status only ``r_i - beta`` is identifiable within a patient, and the
        pre-treatment window of a patient who starts therapy during follow-up
        is the only place the two separate without a between-patient contrast.
    available:
        Optional per-channel boolean mask over ``times``, for missing
        measurements. DLCO in particular goes missing far more often than FVC,
        and often *because* the patient could not complete the breath-hold -
        which makes it missing-not-at-random. This class only represents the
        pattern; it does not fix the bias.
    noise_sd:
        Pinned measurement standard deviations per channel.
    """

    times: np.ndarray
    channels: tuple[Channel, ...] = (Channel.FVC, Channel.DLCO)
    treatment_start: float | None = None
    available: dict[str, np.ndarray] = field(default_factory=dict)
    noise_sd: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_NOISE_SD))

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        if self.times.size == 0 or self.times[0] != 0.0:
            raise ValueError("times must be non-empty and start at 0.")
        if not np.all(np.isfinite(self.times)):
            raise ValueError("times must all be finite.")
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("times must be strictly increasing.")
        if not self.channels:
            raise ValueError("at least one observation channel is required.")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("channels contains duplicates.")

        if self.treatment_start is not None:
            start = float(self.treatment_start)
            if not np.isfinite(start) or start < 0.0:
                raise ValueError(
                    "treatment_start must be a finite, non-negative number of "
                    "years, or None for never treated."
                )
            if start > self.times[-1]:
                # Silently accepting this would report a design as though it
                # had a treatment change, when in fact therapy starts after the
                # last visit and the patient is untreated throughout - which is
                # precisely the design in which beta is not estimable.
                raise ValueError(
                    f"treatment_start ({start:.3f} y) is after the last visit "
                    f"({self.times[-1]:.3f} y); the patient is untreated for "
                    "the whole of this design. Pass treatment_start=None to "
                    "say that explicitly."
                )
            self.treatment_start = start
        for channel in self.channels:
            mask = self.available.get(channel.value)
            if mask is None:
                self.available[channel.value] = np.ones(self.times.shape, bool)
            else:
                mask = np.asarray(mask, dtype=bool)
                if mask.shape != self.times.shape:
                    raise ValueError(
                        f"availability mask for {channel.value} has shape "
                        f"{mask.shape}, expected {self.times.shape}"
                    )
                self.available[channel.value] = mask
            if channel.value not in self.noise_sd:
                raise ValueError(f"no noise_sd given for channel {channel.value}")
            sd = self.noise_sd[channel.value]
            if not np.isfinite(sd) or sd <= 0:
                raise ValueError(
                    f"noise_sd for {channel.value} must be finite and positive"
                )

        if self.n_observations == 0:
            raise ValueError(
                "this design has no available observations at all; every "
                "channel is masked out."
            )

    @property
    def n_observations(self) -> int:
        return int(sum(self.available[c.value].sum() for c in self.channels))

    def stack(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        """Flatten a channel dict to the observed vector, dropping missing."""
        return np.concatenate(
            [
                np.asarray(observations[c.value])[self.available[c.value]]
                for c in self.channels
            ]
        )

    def noise_vector(self) -> np.ndarray:
        """Standard deviation for each entry of the stacked observation vector."""
        return np.concatenate(
            [
                np.full(int(self.available[c.value].sum()), self.noise_sd[c.value])
                for c in self.channels
            ]
        )

    def describe(self) -> str:
        months = ", ".join(f"{t * 12:.0f}" for t in self.times)
        if self.treatment_start is None:
            treatment = "never treated"
        elif self.treatment_start <= 0.0:
            treatment = "treated from baseline"
        else:
            treatment = f"antifibrotic started at month {self.treatment_start * 12:.0f}"
        counts = ", ".join(
            f"{c.value}={int(self.available[c.value].sum())}/{self.times.size}"
            for c in self.channels
        )
        return f"visits at months [{months}]; {treatment}; {counts}"


def routine_followup(
    n_visits: int = 4,
    interval_months: float = 6.0,
    treatment_start: float | None = None,
    channels: tuple[Channel, ...] = (Channel.FVC, Channel.DLCO),
    dlco_missing_rate: float = 0.0,
    seed: int = 0,
) -> VisitSchedule:
    """A routine IPF follow-up schedule, optionally with DLCO dropped at random.

    ``dlco_missing_rate`` produces *missing completely at random* DLCO, which is
    the optimistic case. Real DLCO missingness is informative; use this to study
    the loss of precision from having fewer measurements, not to estimate the
    bias from why they are absent.
    """
    if n_visits < 1:
        raise ValueError(f"n_visits must be at least 1, got {n_visits}")
    if not np.isfinite(interval_months) or interval_months <= 0.0:
        raise ValueError(
            f"interval_months must be finite and positive, got {interval_months}"
        )
    if not 0.0 <= dlco_missing_rate <= 1.0:
        # Out of range values do not fail loudly: a rate above 1 masks every
        # follow-up DLCO and leaves only the baseline, which still analyses
        # cleanly and silently answers a different design question.
        raise ValueError(
            "dlco_missing_rate must lie in [0, 1], got "
            f"{dlco_missing_rate}"
        )

    times = np.arange(n_visits, dtype=float) * (interval_months / 12.0)
    available: dict[str, np.ndarray] = {}
    if dlco_missing_rate > 0.0 and Channel.DLCO in channels:
        rng = np.random.default_rng(seed)
        mask = rng.random(times.shape) >= dlco_missing_rate
        mask[0] = True  # baseline DLCO anchors dlco0
        available[Channel.DLCO.value] = mask
    return VisitSchedule(
        times=times,
        channels=channels,
        treatment_start=treatment_start,
        available=available,
    )

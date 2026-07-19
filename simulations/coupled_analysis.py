"""
The two point-of-no-return analyses, joined.

Two separate bistabilities were found earlier and never tested together:

  * an **epithelial** one, in which profibrotic signal keeps alveolar epithelium
    trapped in the KRT8+ transitional state, so aberrant basaloid cells
    accumulate and sustain their own production;
  * a **matrix** one, in which myofibroblasts stiffen the ECM, and stiffness
    activates more myofibroblasts.

Each was shown to be bistable on its own. That immediately raises the question
this module exists to answer: are they two switches in series, so that breaking
either one resolves the lesion, or does the pair sustain itself through the
cross-couplings even when one loop is cut?

Three coupled variables, all reduced from the agent models in ``alveolar``:

    A - fraction of epithelium held in the aberrant basaloid state
    M - myofibroblast density, normalised to the carrying capacity
    E - local matrix stiffness (kPa)

    dA/dt = k_stall * (1 + beta*P) * K  -  (k_clear + k_emt) * A
    dM/dt = eta * k_emt * A  +  r * (1 - M) * sigma(E)  -  delta * M
    dE/dt = k_dep * M * (1 - E/E_max)  -  k_deg * (E - E_healthy)

    P     = A  +  lambda * (E - E_healthy) / (E_act - E_healthy)
    sigma = 1 / (1 + exp(-(E - E_act)/w))

``P`` is the profibrotic signal. Its first term is secretion by aberrant
epithelium; its second is the stretch-activated TGF-beta released on stiffened
matrix, which is how the mesenchyme talks back to the epithelium. ``K`` is the
transitional (KRT8+) pool feeding the branch point, taken as quasi-steady.

The loops are then explicitly separable:

    beta   epithelial self-promotion   (A -> P -> A)
    sigma  matrix self-promotion       (M -> E -> M)
    eta    epithelium drives mesenchyme (A -> M),  via EMT
    lambda mesenchyme drives epithelium (E -> P),  via stretch-activated TGF-beta

Setting any one of them to zero cuts exactly one arrow, which is what makes the
interruption test meaningful.

This is a reduction, and it discards everything spatial: confinement to the
interstitium, alveolar collapse providing room, durotaxis, the nematic texture.
Its purpose is to locate thresholds cheaply and to say which loop carries them,
after which the agent model is the thing to confirm against.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass
class CoupledParameters:
    """Reduced-model parameters, in units of 1/h and kPa."""

    # --- epithelial loop ---
    krt8_pool: float = 0.35              # K, quasi-steady transitional fraction
    k_stall: float = 0.0015              # KRT8+ -> aberrant
    beta: float = 25.0                   # profibrotic promotion of stalling
    k_clear: float = 0.0005              # clearance (apoptosis resistance is low)
    k_emt: float = 0.004                 # aberrant -> mesenchyme

    # --- mesenchymal / matrix loop ---
    eta: float = 3.0                     # mesenchymal yield per unit EMT flux
    r_activate: float = 0.08             # stiffness-driven activation of residents
    E_act_kPa: float = 16.0
    activation_width_kPa: float = 2.0
    delta_death: float = 0.0004          # myofibroblast death (resistance = small)
    k_dep: float = 0.16                  # collagen deposition
    k_deg: float = 0.004                 # matrix turnover
    E_healthy_kPa: float = 2.0
    E_max_kPa: float = 100.0

    # --- cross-coupling from matrix back to epithelium ---
    lam: float = 0.8                     # stretch-activated TGF-beta per unit stiffness

    # --- scenario ---
    injury_boost: float = 8.0            # multiplies k_stall while the insult lasts
    injury_duration_h: float = 2400.0
    rate_scale: float = 0.08

    @classmethod
    def from_alveolar(cls, config) -> "CoupledParameters":
        """Map an ``AlveolarConfig`` onto the reduced parameters."""
        return cls(
            k_stall=config.krt8_to_aberrant_rate,
            beta=config.stall_promotion_strength,
            k_clear=config.aberrant_clearance_rate,
            k_emt=config.aberrant_emt_rate,
            r_activate=config.activation_rate_per_h,
            E_act_kPa=config.E_act_kPa,
            activation_width_kPa=config.activation_width_kPa,
            delta_death=config.myofibroblast_death_rate,
            k_dep=config.deposition_rate_kPa_per_h,
            k_deg=config.degradation_rate_per_h,
            E_healthy_kPa=config.E_healthy_kPa,
            E_max_kPa=config.E_max_kPa,
            lam=config.strain_tgfb_gain,
            injury_duration_h=config.injury_duration_h,
            rate_scale=config.rate_scale,
        )

    def scaled(self) -> "CoupledParameters":
        """Apply the global clock, exactly as the agent model does."""
        if self.rate_scale == 1.0:
            return self
        # Matches the agent model: only matrix and disease kinetics are slowed.
        # Cell death and activation keep their own clock.
        names = ("k_stall", "k_clear", "k_emt", "r_activate", "k_dep", "k_deg")
        changes = {n: getattr(self, n) * self.rate_scale for n in names}
        out = replace(self, **changes)
        out.rate_scale = 1.0
        return out


def profibrotic(A: np.ndarray, E: np.ndarray, p: CoupledParameters) -> np.ndarray:
    span = max(p.E_act_kPa - p.E_healthy_kPa, 1e-9)
    return A + p.lam * np.clip((E - p.E_healthy_kPa) / span, 0.0, None)


def activation(E: np.ndarray, p: CoupledParameters) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(E - p.E_act_kPa) / p.activation_width_kPa))


def velocity(state: np.ndarray, p: CoupledParameters,
             injured: bool = False) -> np.ndarray:
    """Right-hand side [dA/dt, dM/dt, dE/dt]."""
    A, M, E = state
    P = profibrotic(A, E, p)
    stall = p.k_stall * (p.injury_boost if injured else 1.0)

    dA = stall * (1.0 + p.beta * P) * p.krt8_pool - (p.k_clear + p.k_emt) * A
    dA -= stall * (1.0 + p.beta * P) * p.krt8_pool * A      # saturation at A -> 1
    dM = (p.eta * p.k_emt * A
          + p.r_activate * (1.0 - M) * activation(E, p)
          - p.delta_death * M)
    dE = p.k_dep * M * (1.0 - E / p.E_max_kPa) - p.k_deg * (E - p.E_healthy_kPa)
    return np.array([dA, dM, dE])


def integrate(p: CoupledParameters, total_time_h: float = 26280.0,
              dt_h: float = 2.0, state0: np.ndarray | None = None
              ) -> dict:
    """Run through the insult and its withdrawal; report where it settles."""
    q = p.scaled()
    state = (np.array([0.0, 0.0, q.E_healthy_kPa])
             if state0 is None else np.asarray(state0, dtype=float))
    n_steps = int(round(total_time_h / dt_h))
    trace = np.empty((n_steps + 1, 3))
    trace[0] = state
    for index in range(1, n_steps + 1):
        injured = (index * dt_h) < q.injury_duration_h
        state = state + dt_h * velocity(state, q, injured)
        state[0] = float(np.clip(state[0], 0.0, 1.0))
        state[1] = float(np.clip(state[1], 0.0, 1.0))
        state[2] = float(np.clip(state[2], q.E_healthy_kPa, q.E_max_kPa))
        trace[index] = state
    return {
        "A_final": float(state[0]),
        "M_final": float(state[1]),
        "E_final_kPa": float(state[2]),
        "fibrotic": bool(state[2] > q.E_act_kPa),
        "trace": trace,
    }


def _M_steady(A: float | np.ndarray, E: float | np.ndarray,
              p: CoupledParameters):
    """Myofibroblast density at steady state for given A and E (solved exactly)."""
    sigma = activation(np.asarray(E, dtype=float), p)
    numerator = p.eta * p.k_emt * np.asarray(A, dtype=float) + p.r_activate * sigma
    denominator = p.r_activate * sigma + p.delta_death
    return np.clip(numerator / np.maximum(denominator, 1e-30), 0.0, 1.0)


def fixed_points(p: CoupledParameters, n_grid: int = 400) -> list[dict]:
    """Steady states, found on an (A, E) grid with M eliminated analytically.

    ``dM/dt = 0`` can be solved in closed form for M, which turns the problem
    into two curves in the (A, E) plane. Their intersections are the fixed
    points. This replaces multidimensional root finding, which failed to
    converge on this system.
    """
    q = p.scaled()
    A_axis = np.linspace(0.0, 1.0, n_grid)
    E_axis = np.linspace(q.E_healthy_kPa, q.E_max_kPa, n_grid)
    A, E = np.meshgrid(A_axis, E_axis, indexing="ij")
    M = _M_steady(A, E, q)

    P = profibrotic(A, E, q)
    dA = (q.k_stall * (1.0 + q.beta * P) * q.krt8_pool * (1.0 - A)
          - (q.k_clear + q.k_emt) * A)
    dE = q.k_dep * M * (1.0 - E / q.E_max_kPa) - q.k_deg * (E - q.E_healthy_kPa)

    # a cell where both residuals change sign contains an intersection
    sa = np.sign(dA); se = np.sign(dE)
    flip_a = (sa[:-1, :-1] * sa[1:, :-1] <= 0) | (sa[:-1, :-1] * sa[:-1, 1:] <= 0)
    flip_e = (se[:-1, :-1] * se[1:, :-1] <= 0) | (se[:-1, :-1] * se[:-1, 1:] <= 0)
    rows, cols = np.nonzero(flip_a & flip_e)

    results: list[dict] = []
    for i, j in zip(rows, cols):
        state = np.array([A_axis[i], _M_steady(A_axis[i], E_axis[j], q),
                          E_axis[j]])
        if any(abs(state[0] - r["A"]) < 0.02 and abs(state[2] - r["E_kPa"]) < 1.0
               for r in results):
            continue
        eigenvalues = np.linalg.eigvals(_jacobian(state, q))
        results.append({
            "A": float(state[0]), "M": float(state[1]), "E_kPa": float(state[2]),
            "stable": bool(np.all(eigenvalues.real < 1e-12)),
            "max_real_eigenvalue": float(eigenvalues.real.max()),
        })
    return sorted(results, key=lambda r: r["E_kPa"])


def _jacobian(state: np.ndarray, p: CoupledParameters,
              epsilon: float = 1e-6) -> np.ndarray:
    jacobian = np.zeros((3, 3))
    base = velocity(state, p)
    for index in range(3):
        shifted = state.copy()
        shifted[index] += epsilon
        jacobian[:, index] = (velocity(shifted, p) - base) / epsilon
    return jacobian


def loop_interruption_test(p: CoupledParameters, **integrate_kwargs) -> pd.DataFrame:
    """Cut each arrow in turn and see whether the lesion still becomes fibrotic.

    If the two switches were simply in series, breaking any single loop would
    resolve the lesion. Any row that stays fibrotic identifies a path the
    disease can still take on its own.
    """
    cases = {
        "intact": {},
        "epithelial loop cut (beta=0)": {"beta": 0.0},
        "matrix loop cut (r_activate=0)": {"r_activate": 0.0},
        "EMT link cut (eta=0)": {"eta": 0.0},
        "stretch TGF-beta cut (lambda=0)": {"lam": 0.0},
        "both loops cut": {"beta": 0.0, "r_activate": 0.0},
        "both links cut": {"eta": 0.0, "lam": 0.0},
        "clearance restored (x20)": {"k_clear": p.k_clear * 20},
        "turnover restored (x5)": {"k_deg": p.k_deg * 5},
    }
    rows = []
    for label, changes in cases.items():
        variant = replace(p, **changes) if changes else p
        result = integrate(variant, **integrate_kwargs)
        rows.append({
            "intervention": label,
            "A_final": result["A_final"],
            "M_final": result["M_final"],
            "E_final_kPa": result["E_final_kPa"],
            "fibrotic": result["fibrotic"],
        })
    return pd.DataFrame(rows)


def phase_scan(p: CoupledParameters, x_name: str, x_values: np.ndarray,
               y_name: str, y_values: np.ndarray, **integrate_kwargs
               ) -> pd.DataFrame:
    """Outcome over a grid of any two reduced parameters."""
    rows = []
    for x in x_values:
        for y in y_values:
            variant = replace(p, **{x_name: float(x), y_name: float(y)})
            result = integrate(variant, **integrate_kwargs)
            rows.append({
                x_name: float(x), y_name: float(y),
                "E_final_kPa": result["E_final_kPa"],
                "A_final": result["A_final"],
                "fibrotic": result["fibrotic"],
            })
    return pd.DataFrame(rows)

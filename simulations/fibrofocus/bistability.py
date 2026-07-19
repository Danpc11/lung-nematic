"""
Where is the point of no return?

The agent simulation shows *what* a focus looks like; this module says *when*
one becomes irreversible. It reduces the same biology to a single equation for
the local stiffness E of the lesion:

    dE/dt = k_dep * f(E) * (1 - E/E_max) - k_deg * (E - E_healthy)

where ``f(E)`` is the steady-state myofibroblast fraction produced by the same
stiffness-gated switch used by the agents, including its hysteresis:

    on(E)  = a / (1 + exp(-(E - E_act) / w))
    off(E) = b / (1 + exp(+(E - E_act/memory) / w))
    f(E)   = on / (on + off)

Because ``f`` is switch-like and ``a >> b`` (activation is fast, reversion is
slow: mechanical memory), ``dE/dt`` can have three roots:

    E_healthy  <  E_separatrix  <  E_fibrotic
      stable        UNSTABLE        stable

``E_separatrix`` is the point of no return. A lesion pushed above it becomes
self-sustaining after the epithelial insult is withdrawn; below it, the lesion
resolves. When only one root exists the system is monostable and no persistent
focus can form, whatever the insult.

This reduction is deliberately coarse: it ignores the spatial structure,
durotactic accumulation and the nematic mechanics that the agent model
resolves. Its purpose is to locate thresholds cheaply so the expensive agent
runs can be aimed at the interesting places.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .model import FocusConfig


def myofibroblast_fraction(E: np.ndarray, cfg: FocusConfig) -> np.ndarray:
    """Steady-state activated fraction at stiffness E (with memory)."""
    w = cfg.activation_width_kPa
    E_deact = cfg.E_act_kPa / cfg.memory_factor
    on = cfg.activation_rate_per_h / (1.0 + np.exp(-(E - cfg.E_act_kPa) / w))
    off = cfg.deactivation_rate_per_h / (1.0 + np.exp((E - E_deact) / w))
    return on / np.maximum(on + off, 1e-300)


def stiffness_velocity(E: np.ndarray, cfg: FocusConfig,
                       myo_supply: float = 1.0) -> np.ndarray:
    """dE/dt for the reduced model.

    ``myo_supply`` scales the available myofibroblast density relative to the
    carrying density, i.e. how strongly durotaxis and proliferation concentrate
    cells in the lesion.
    """
    f = myofibroblast_fraction(E, cfg) * myo_supply
    deposition = cfg.deposition_rate_kPa_per_h * f * (1.0 - E / cfg.E_max_kPa)
    degradation = cfg.degradation_rate_per_h * (E - cfg.E_healthy_kPa)
    return deposition - degradation


def fixed_points(cfg: FocusConfig, myo_supply: float = 1.0,
                 n_samples: int = 4000) -> dict:
    """Locate the roots of dE/dt and classify the system."""
    E = np.linspace(cfg.E_healthy_kPa, cfg.E_max_kPa, n_samples)
    v = stiffness_velocity(E, cfg, myo_supply)

    roots: list[float] = []
    stable: list[bool] = []
    for index in range(len(E) - 1):
        if v[index] == 0.0 or v[index] * v[index + 1] < 0:
            # linear interpolation of the crossing
            if v[index + 1] != v[index]:
                frac = v[index] / (v[index] - v[index + 1])
            else:
                frac = 0.0
            root = E[index] + frac * (E[index + 1] - E[index])
            roots.append(float(root))
            stable.append(bool(v[index] > v[index + 1]))

    # the boundary E = E_healthy is a stable state when dE/dt <= 0 there
    if not roots or roots[0] > cfg.E_healthy_kPa + 1e-9:
        if v[0] <= 0:
            roots.insert(0, float(cfg.E_healthy_kPa))
            stable.insert(0, True)

    stable_roots = [r for r, s in zip(roots, stable) if s]
    unstable_roots = [r for r, s in zip(roots, stable) if not s]

    bistable = len(stable_roots) >= 2 and len(unstable_roots) >= 1
    return {
        "roots": roots,
        "stable": stable_roots,
        "unstable": unstable_roots,
        "bistable": bistable,
        "E_separatrix": float(unstable_roots[0]) if unstable_roots else float("nan"),
        "E_fibrotic": float(max(stable_roots)) if stable_roots else float("nan"),
        "E_healthy": float(min(stable_roots)) if stable_roots else float("nan"),
    }


def integrate_lesion(cfg: FocusConfig, myo_supply: float = 1.0,
                     E_start: float | None = None,
                     dt_h: float | None = None) -> dict:
    """Integrate the reduced model through insult and withdrawal.

    During the insult the lesion is driven toward the provisional-matrix
    stiffness; afterwards it evolves freely. Returns whether the lesion
    persisted. ``dt_h`` overrides the agent-model timestep: the reduced
    equation is smooth, so a coarser step is accurate and much faster for
    parameter scans.
    """
    dt = cfg.dt_h if dt_h is None else float(dt_h)
    n_steps = int(round(cfg.total_time_h / dt))
    E = cfg.E_healthy_kPa if E_start is None else float(E_start)
    trace = np.empty(n_steps + 1)
    trace[0] = E
    for step in range(1, n_steps + 1):
        t = step * dt
        dE = stiffness_velocity(np.array([E]), cfg, myo_supply)[0]
        if t < cfg.injury_duration_h:
            dE += cfg.injury_stiffening_rate_per_h * max(
                cfg.injury_provisional_E_kPa - E, 0.0
            )
        E = float(np.clip(E + dt * dE, cfg.E_healthy_kPa, cfg.E_max_kPa))
        trace[step] = E

    info = fixed_points(cfg, myo_supply)
    persisted = bool(E > cfg.E_act_kPa)
    return {
        "E_final_kPa": E,
        "persisted": persisted,
        "E_separatrix": info["E_separatrix"],
        "bistable": info["bistable"],
        "trace": trace,
    }


def scan_two_parameters(
    base: FocusConfig,
    x_name: str,
    x_values: np.ndarray,
    y_name: str,
    y_values: np.ndarray,
    myo_supply: float = 1.0,
    dt_h: float | None = 0.5,
) -> pd.DataFrame:
    """Phase diagram over any two config fields.

    For every combination the reduced model is integrated through the insult
    and its withdrawal, recording whether the lesion becomes persistent and
    where the separatrix sits.
    """
    rows: list[dict] = []
    for x in x_values:
        for y in y_values:
            cfg = replace(base, **{x_name: float(x), y_name: float(y)})
            cfg.validate()
            result = integrate_lesion(cfg, myo_supply, dt_h=dt_h)
            rows.append(
                {
                    x_name: float(x),
                    y_name: float(y),
                    "E_final_kPa": result["E_final_kPa"],
                    "persisted": result["persisted"],
                    "bistable": result["bistable"],
                    "E_separatrix": result["E_separatrix"],
                }
            )
    return pd.DataFrame(rows)


def critical_value(
    base: FocusConfig,
    name: str,
    low: float,
    high: float,
    myo_supply: float = 1.0,
    tolerance: float = 1e-4,
    max_iterations: int = 60,
    dt_h: float | None = 0.5,
) -> float:
    """Bisect for the critical value of one parameter (the no-return point).

    Assumes persistence is monotonic in the parameter between ``low`` and
    ``high``. Returns NaN when both ends give the same outcome.
    """
    def persists(value: float) -> bool:
        cfg = replace(base, **{name: float(value)})
        cfg.validate()
        return integrate_lesion(cfg, myo_supply, dt_h=dt_h)["persisted"]

    low_state, high_state = persists(low), persists(high)
    if low_state == high_state:
        return float("nan")

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        if persists(mid) == low_state:
            low = mid
        else:
            high = mid
        if abs(high - low) < tolerance * max(1.0, abs(high)):
            break
    return float(0.5 * (low + high))

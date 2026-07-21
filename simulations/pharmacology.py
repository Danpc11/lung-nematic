"""
Retrospective pharmacological validation.

A mechanistic model of fibrosis will rank almost any implicated target as
treatable, because every implicated target is, by construction, load-bearing
inside the model. That makes agreement with biology worthless as evidence. The
only discriminating test is whether the model reproduces interventions that
*failed* in patients despite excellent mechanistic rationale.

The controls below are therefore split not by mechanism but by outcome:

**Slows progression in patients** — nintedanib and pirfenidone both reduce the
rate of FVC decline by roughly half without halting or reversing disease. A
model that predicts either one *cures* the lesion is wrong in the same way a
model that predicts no effect is wrong.

**Effective preclinically, untested or unproven clinically** — YAP/TAZ and
FAK/ROCK inhibition. These are held to a weaker standard: the model should show
benefit, but agreement carries little evidential weight.

**Failed in patients** — the discriminating cases.

  * *LOXL2 (simtuzumab).* Blocking collagen crosslinking had a compelling
    rationale and convincing rodent data. The phase 2 trial was terminated for
    futility, and later translational work found the antibody *promoted*
    fibroblast-to-myofibroblast transition and increased invasion of IPF
    fibroblasts rather than being merely inert.
  * *alpha-v integrin blockade.* Frequently listed as a positive control, but
    BG00011/STX-100 was terminated early in phase 2 on safety grounds,
    GSK3008348 was discontinued after phase 1b, and pan-alpha-v IDL-2965 was
    terminated. Only the dual alpha-v-beta-1/beta-6 inhibitor bexotegrast has
    positive phase 2 data, so single-target alpha-v-beta-6 blockade belongs
    with the failures.

A model that resolves the lesion for either of the last two is overfitted to
biological plausibility. The point of this module is to make that failure
visible rather than to avoid it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .coupled_analysis import CoupledParameters, integrate


@dataclass(frozen=True)
class Intervention:
    """One drug or target, mapped onto reduced-model parameters."""

    name: str
    target: str
    # multiplicative changes applied to CoupledParameters fields
    effects: dict
    expected: str            # "slows" | "fails" | "preclinical_benefit"
    evidence: str

    def apply(self, parameters: CoupledParameters) -> CoupledParameters:
        changes = {
            name: getattr(parameters, name) * factor
            for name, factor in self.effects.items()
        }
        return replace(parameters, **changes)


# Each effect is a multiplier on the named reduced-model parameter. The
# strength is deliberately generous (a large, not marginal, pharmacological
# effect) so that a model failing to reproduce a clinical failure cannot blame
# an under-dosed intervention.
INTERVENTIONS: tuple[Intervention, ...] = (
    Intervention(
        name="nintedanib",
        target="PDGFR / FGFR / VEGFR",
        effects={"eta": 0.4, "r_activate": 0.6},
        expected="slows",
        evidence="reduces annual FVC decline by roughly half; does not halt or reverse",
    ),
    Intervention(
        name="pirfenidone",
        target="multi-pathway, incl. collagen synthesis",
        effects={"k_dep": 0.5},
        expected="slows",
        evidence="reduces annual FVC decline by roughly half; does not halt or reverse",
    ),
    Intervention(
        name="YAP/TAZ inhibition",
        target="mechanotransduction to the nucleus",
        effects={"r_activate": 0.1},
        expected="preclinical_benefit",
        evidence="preclinical only; no completed IPF efficacy trial",
    ),
    Intervention(
        name="FAK/ROCK inhibition",
        target="adhesion and contractility",
        effects={"r_activate": 0.25, "k_dep": 0.5},
        expected="preclinical_benefit",
        evidence="preclinical only",
    ),
    Intervention(
        name="alpha-v-beta-6 blockade",
        target="mechanical activation of latent TGF-beta",
        effects={"lam": 0.05},
        expected="fails",
        evidence="BG00011/STX-100 terminated early in phase 2; GSK3008348 discontinued",
    ),
    Intervention(
        name="LOXL2 (simtuzumab)",
        target="collagen crosslinking",
        effects={"k_deg": 5.0},
        expected="fails",
        evidence="phase 2 terminated for futility; later shown to promote myofibroblast transition",
    ),
)


def classify_response(
    treated_E: float,
    untreated_E: float,
    healthy_E: float,
    activation_E: float,
) -> str:
    """Turn a stiffness outcome into a clinically comparable category."""
    if treated_E <= activation_E:
        return "resolves"
    span = max(untreated_E - healthy_E, 1e-9)
    reduction = (untreated_E - treated_E) / span
    if reduction >= 0.15:
        return "slows"
    return "no_benefit"


def run_panel(
    parameters: CoupledParameters,
    interventions: tuple[Intervention, ...] = INTERVENTIONS,
    **integrate_kwargs,
) -> pd.DataFrame:
    """Run every control and score it against the clinical record."""
    baseline = integrate(parameters, **integrate_kwargs)
    untreated_E = baseline["E_final_kPa"]
    scaled = parameters.scaled()

    rows = [
        {
            "intervention": "untreated",
            "target": "-",
            "A_final": baseline["A_final"],
            "E_final_kPa": untreated_E,
            "response": "reference",
            "expected": "-",
            "agrees": True,
            "evidence": "-",
        }
    ]

    for item in interventions:
        result = integrate(item.apply(parameters), **integrate_kwargs)
        response = classify_response(
            result["E_final_kPa"], untreated_E,
            scaled.E_healthy_kPa, scaled.E_act_kPa,
        )
        if item.expected == "slows":
            agrees = response == "slows"
        elif item.expected == "fails":
            agrees = response == "no_benefit"
        else:                                   # preclinical_benefit
            agrees = response in ("slows", "resolves")

        rows.append(
            {
                "intervention": item.name,
                "target": item.target,
                "A_final": result["A_final"],
                "E_final_kPa": result["E_final_kPa"],
                "response": response,
                "expected": item.expected,
                "agrees": bool(agrees),
                "evidence": item.evidence,
            }
        )

    return pd.DataFrame(rows)


def score(panel: pd.DataFrame) -> dict:
    """Summarise a panel, weighting the clinical failures most heavily.

    Agreement with a target that worked proves little; a model is only
    discriminating if it also reproduces the failures.
    """
    scored = panel.loc[panel["expected"] != "-"]
    failures = scored.loc[scored["expected"] == "fails"]
    return {
        "n_controls": int(len(scored)),
        "n_agree": int(scored["agrees"].sum()),
        "clinical_failures_reproduced": int(failures["agrees"].sum()),
        "clinical_failures_total": int(len(failures)),
        "passes_discriminating_test": bool(failures["agrees"].all()),
    }


# ---------------------------------------------------------------------------
# The structural fix the LOXL2 control forces
# ---------------------------------------------------------------------------
#
# A single lumped stiffness conflates three different things: how much collagen
# is present, how crosslinked it is, and how degradable that makes it. Raising
# one turnover rate therefore dissolves established scar, which is why the model
# above declares LOXL2 inhibition a cure.
#
# Crosslinking is not reversible by blocking the enzyme that creates it. LOXL2
# inhibition stops *new* crosslinks forming; it cannot un-crosslink matrix that
# is already mature. Splitting the matrix into a newly deposited, still
# degradable compartment and a crosslinked, effectively permanent one makes the
# intervention act only on the flux between them:
#
#     dE_new/dt    = deposition - k_deg * E_new - k_mature * E_new
#     dE_mature/dt = k_mature * E_new - k_deg_mature * E_mature
#
# The prediction that follows is not fitted, it is forced: blocking crosslinking
# helps when little mature matrix exists yet, and does almost nothing once the
# lesion is established. That is exactly the pattern in the record - efficacy in
# preventative rodent bleomycin models, futility in patients with established
# fibrosis.

@dataclass(frozen=True)
class MaturationParameters:
    """Matrix split into degradable and crosslinked compartments."""

    k_mature: float = 0.004          # crosslinking flux, new -> mature
    k_deg_mature: float = 0.00015    # crosslinked matrix barely turns over


def integrate_with_maturation(
    parameters: CoupledParameters,
    maturation: MaturationParameters = MaturationParameters(),
    total_time_h: float = 26280.0,
    dt_h: float = 2.0,
    treated: CoupledParameters | None = None,
    treated_maturation: MaturationParameters | None = None,
    treatment_start_h: float = 0.0,
) -> dict:
    """Integrate with the matrix split into new and crosslinked compartments.

    ``treatment_start_h`` is the distinction between a prevention study and a
    treatment study, and it is not cosmetic. Rodent bleomycin work that
    supported LOXL2 blockade dosed *before* fibrosis was established; the phase
    2 trial enrolled patients who already had it. An intervention that acts on
    the flux into a permanent compartment can only help while that compartment
    is still filling, so the two designs are not expected to agree.
    """
    from .coupled_analysis import activation, profibrotic

    baseline_q = parameters.scaled()
    treated_q = (treated or parameters).scaled()
    scale = parameters.rate_scale
    active_maturation = treated_maturation or maturation

    A, M = 0.0, 0.0
    new, mature = 0.0, 0.0
    n_steps = int(round(total_time_h / dt_h))

    for index in range(1, n_steps + 1):
        time_h = index * dt_h
        on_drug = time_h >= treatment_start_h
        q = treated_q if on_drug else baseline_q
        current_maturation = active_maturation if on_drug else maturation
        k_mature = current_maturation.k_mature * scale
        k_deg_mature = current_maturation.k_deg_mature * scale

        injured = time_h < q.injury_duration_h
        E = q.E_healthy_kPa + new + mature
        P = profibrotic(np.array(A), np.array(E), q)
        stall = q.k_stall * (q.injury_boost if injured else 1.0)

        dA = (stall * (1.0 + q.beta * float(P)) * q.krt8_pool * (1.0 - A)
              - (q.k_clear + q.k_emt) * A)
        dM = (q.eta * q.k_emt * A
              + q.r_activate * (1.0 - M) * float(activation(np.array(E), q))
              - q.delta_death * M)
        deposition = q.k_dep * M * (1.0 - E / q.E_max_kPa)
        d_new = deposition - q.k_deg * new - k_mature * new
        d_mature = k_mature * new - k_deg_mature * mature

        A = float(np.clip(A + dt_h * dA, 0.0, 1.0))
        M = float(np.clip(M + dt_h * dM, 0.0, 1.0))
        new = float(np.clip(new + dt_h * d_new, 0.0, q.E_max_kPa))
        mature = float(np.clip(mature + dt_h * d_mature, 0.0, q.E_max_kPa))

    E_final = treated_q.E_healthy_kPa + new + mature
    return {
        "A_final": A, "M_final": M,
        "E_final_kPa": min(E_final, q.E_max_kPa),
        "E_new_kPa": new, "E_mature_kPa": mature,
        "fibrotic": bool(E_final > q.E_act_kPa),
    }


LOXL2_MATURATION_EFFECT = 0.05      # blocking crosslinking slows maturation


def run_panel_with_maturation(
    parameters: CoupledParameters,
    maturation: MaturationParameters = MaturationParameters(),
    interventions: tuple[Intervention, ...] = INTERVENTIONS,
    treatment_start_h: float = 0.0,
    **integrate_kwargs,
) -> pd.DataFrame:
    """Same panel, with LOXL2 acting on crosslinking and a dosing start time.

    ``treatment_start_h = 0`` reproduces a prevention study; a start well after
    the lesion is established reproduces a treatment study.
    """
    baseline = integrate_with_maturation(parameters, maturation, **integrate_kwargs)
    untreated_E = baseline["E_final_kPa"]
    scaled = parameters.scaled()

    rows = [{"intervention": "untreated", "E_final_kPa": untreated_E,
             "E_mature_kPa": baseline["E_mature_kPa"], "response": "reference",
             "expected": "-", "agrees": True}]

    for item in interventions:
        if item.name.startswith("LOXL2"):
            # acts on the crosslinking flux, not on bulk degradability
            result = integrate_with_maturation(
                parameters, maturation,
                treated_maturation=replace(
                    maturation,
                    k_mature=maturation.k_mature * LOXL2_MATURATION_EFFECT,
                ),
                treatment_start_h=treatment_start_h, **integrate_kwargs,
            )
        else:
            result = integrate_with_maturation(
                parameters, maturation, treated=item.apply(parameters),
                treatment_start_h=treatment_start_h, **integrate_kwargs,
            )

        response = classify_response(
            result["E_final_kPa"], untreated_E,
            scaled.E_healthy_kPa, scaled.E_act_kPa,
        )
        if item.expected == "slows":
            agrees = response == "slows"
        elif item.expected == "fails":
            agrees = response == "no_benefit"
        else:
            agrees = response in ("slows", "resolves")

        rows.append({"intervention": item.name, "E_final_kPa": result["E_final_kPa"],
                     "E_mature_kPa": result["E_mature_kPa"], "response": response,
                     "expected": item.expected, "agrees": bool(agrees)})

    return pd.DataFrame(rows)

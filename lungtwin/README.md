# lungtwin

Identifiability analysis for a low-dimensional IPF digital twin.

This package is deliberately **not** an estimator. It answers what can be
estimated from a given study design, before any estimator exists. The reason is
that a fit to a structurally unidentifiable model does not fail loudly — the
optimiser returns whatever point the initialisation drifted to, the residuals
look fine, and the result is indistinguishable from an estimate.

## The reparameterized model

```
dB/dt = (r_i − β·T(t)) · f(R)          B(0) = 0     burden, in FVC points
dR/dt = −α·B                           R(0) = 1     reserve, dimensionless

FVC(t)  = FVC₀ − B(t)                               c_F ≡ 1 fixes the scale
DLCO(t) = DLCO₀ − κ·B(t)                            κ = c_D/c_F
SpO₂(t) = SpO₂₀ − λ·(1 − R(t))                      only in OBSERVED mode
```

Two conventions (`c_F ≡ 1`, `B(0) ≡ 0`) remove the scale and offset degeneracies
of the original specification, where only the products `c_F·r_i` and `c_F·β`
were determined and `B₀` traded off against `FVC₀` and `DLCO₀`. What survives is
a single coupling ratio `κ`, which is identifiable and clinically interpretable.

Defaults are population anchors from the PROFILE incident cohort (12-month
change of −5.28 ppFVC and −3.35 ppDLCO, giving `κ ≈ 0.63`) and from the
antifibrotic trials (`β` ≈ 50% reduction in decline). **Note the direction: in
percentage-point terms DLCO falls more slowly than FVC**, the opposite of the
usual intuition.

### The reserve state

In the original specification `R` appeared in neither the burden equation nor
any observation equation. Its sensitivity was identically zero, so `R₀` and `α`
were structurally unidentifiable and deleting the entire state changed no
prediction. `ReserveMode` makes the three resolutions explicit:

| mode | wiring | consequence |
|---|---|---|
| `NONE` | `R` dropped | `B` linear in time → the model reduces algebraically to per-patient linear regression |
| `OBSERVED` | `R` gets an SpO₂ channel | identifiable through data |
| `FEEDBACK` | `R` modulates the burden rate | the mechanosensitive hypothesis; the only mode where mechanism does work a hierarchical linear model could not |

## Findings

Everything below is reproduced by the test suite.

**β is estimable only from a within-patient change of treatment status.**
With a constant status only `r_i − β` is identifiable; the pre-treatment window
of a patient who starts therapy during follow-up is the only place the two
separate without a between-patient contrast, and that contrast carries
confounding by indication.

| design (4 visits, 6-monthly) | rank | null direction |
|---|---|---|
| never treated | 4/5 | `{β}` |
| treated from baseline | 4/5 | `{r_i, β}` — only the difference |
| **starts antifibrotic at 6 mo** | **5/5** | — |
| starts at 6 mo, no DLCO | 3/5 | `{κ}`, `{DLCO₀}` |

**Structural identifiability is not enough.** The switcher design is full rank
but its Cramér–Rao bound gives `SE(r_i) = 7.7` against a true value of 5.28 —
the design cannot distinguish a progressing patient from a stable one.

**Longer follow-up does not fix it.** With β free, `SE(r_i)` plateaus near 6 and
does not fall below 5.9 even at 30 visits over 15 years, while `corr(r_i, β)`
rises to 0.9998. β is informed only by the pre-treatment window, whose length is
fixed by when therapy started; visits after the switch add information about
`r_i − β` but almost none about their separation.

**The pre-treatment window is the binding constraint, not total follow-up.**
Holding follow-up at 4 years and varying when therapy starts:

| therapy starts | SE(`r_i`) | SE(β) |
|---|---|---|
| 3 months | 14.13 | 14.70 |
| 6 months | 6.82 | 7.36 |
| 12 months | 3.46 | 4.28 |
| 24 months | 1.70 | 3.47 |
| 36 months | 1.11 | 7.35 |

**Pinning β makes the design feasible.** Fixing β from the trial literature and
letting the data update it weakly drops `SE(r_i)` to 1.43 at 6 visits and 0.93
at 8. This is not a convenience; it is what makes the model usable.

**The Fisher bound has a blind spot, and this model hits it.** `κ` multiplies
the burden, so it is identifiable only while burden actually accrues. In
replicates whose fitted trajectory is near-flat, `κ` is unconstrained and runs to
arbitrary values at no cost in likelihood: the plain empirical SD exceeded the
CRLB by three orders of magnitude while the robust (MAD-based) SD still agreed
with it. `monte_carlo_check` reports both, and the gap is the diagnostic.
Clinically the result is correct — you cannot estimate the DLCO-to-FVC coupling
in a patient who is not declining — and it means **`κ` must be partially pooled
across patients, not estimated per patient.**

## Usage

```bash
pip install -e .

# The design you probably have
lungtwin-ident --visits 4 --treatment-start-months 6 --target-se-ri 1.0

# The design that works
lungtwin-ident --visits 8 --treatment-start-months 6 --fix-beta \
               --target-se-ri 1.0 --monte-carlo 400
```

```python
from lungtwin import Parameters, ReserveMode, analyze, render, routine_followup

schedule = routine_followup(n_visits=4, treatment_start=0.5)
report = analyze(Parameters(), ["fvc0", "dlco0", "r_i", "kappa", "beta"],
                 schedule, ReserveMode.NONE)
print(render(report))
```

## What this package does not do

**Measurement noise is pinned, not estimated.** With three or four visits,
measurement error and process noise both produce scatter around the trajectory
and are not jointly identifiable. `DEFAULT_NOISE_SD` holds stand-in
repeatability values; **replace them with your own laboratory's.**

**Informative dropout is not modelled.** Patients who progress fastest stop
having visits — death, transplant. In PROFILE, 48% met progression criteria at
one year and median survival was 3.7 years. Training on observed visits without
modelling the dropout systematically underestimates `r_i`, and this is the most
serious threat to external validity. It needs a joint longitudinal-survival
model, or at minimum an IPW sensitivity analysis. `dlco_missing_rate` generates
*missing completely at random* DLCO, which is the optimistic case; it models the
precision loss, never the bias.

**The reserve feedback is not testable at routine horizons.** Over 18 months the
difference between `γ = 0` and `γ = 1` is below a third of the FVC measurement
SD — asserted as a test so nobody sells it as an empirical claim at that
horizon. Include the nonlinearity as a mechanistically motivated regulariser;
defer its test to 24–36 months.

**Nothing here consumes `lung-nematic`.** The two-state model uses no imaging
input. That is correct sequencing, not an oversight, and it should not be
described as though the imaging package supplies the mechanism. The intended
linkage is later and narrow: baseline defect density as a prior on `r_i`.

**`P(true decline ≥ x) ≠ P(observed decline ≥ x)`.** Any threshold-crossing
probability must say which one it is. The latter is inflated by spirometer
noise. The conventional progression criterion is a relative decline of ≥10%, not
an absolute 5 points.

**Baselines for any eventual estimator.** Last-observation-carried-forward and
per-patient OLS are weak comparators; per-patient OLS *is* this model under
`ReserveMode.NONE`. The real competitor is a linear mixed model with random
intercept and slope, evaluated by prospective landmark prediction — fit on
visits up to `t`, predict `t + 6mo`, repeated over `t`.

**OSIC cannot validate the full model.** It has 176 patients with FVC over ~1.5
years, plus age, sex and smoking status — no DLCO and no treatment status. Per
the rank table above that is the 3/5 case: neither `κ` nor `β` is identifiable
there. It is a good bench for the filter and for uncertainty calibration, not
for the full model.

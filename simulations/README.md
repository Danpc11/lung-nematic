# Simulations / alveolar-PF

Stage 1+2 of the pulmonary fibrosis model: **alveolar architecture and the
epithelial state machine**. This is the layer that was missing from the
mesenchymal (fibroblastic focus) simulation — it supplies the lesion that
starts everything, and the mesenchymal cells that the focus is built from.

```
simulations/alveolar/
├── geometry.py   Voronoi alveoli, septa, epithelial segments
├── model.py      state machine, surfactant, collapse, induration
└── render.py     frames, GIF and MP4
```

## What it models

The alveolar region is a Voronoi tessellation of a jittered hexagonal lattice:
each cell is an alveolus (~200 µm), each shared ridge an interalveolar septum.
Septa are cut into ~12 µm segments, and every segment carries one epithelial
state:

```
empty (denuded) → AT2 → KRT8+ transitional → AT1        (repair succeeds)
                              ↓
                    aberrant basaloid → EMT → mesenchyme (repair fails)
```

**The lesion is a differentiation failure, not a wound.** Inside the injury
region, AT2 cells are driven to attempt differentiation (`injury_activation_boost`)
while the exit to AT1 is suppressed (`repair_failure_factor`). Neither on its
own does anything: driving without blocking just cycles cells through the
transitional state; blocking without driving leaves it empty. Together they
create a reservoir of KRT8+ cells that can stall.

**Mechanics closes the loop.** AT2 cells produce surfactant; surfactant sets
alveolar surface tension; the Laplace pressure `2γ/r` is opposed by tissue
recoil. When AT2 cells are consumed by the transitional state, surfactant
falls, and alveoli derecruit — smallest first, because `2γ/r` is largest there.
Collapsed alveoli suffer faster epithelial damage and produce less surfactant,
and after `induration_time_h` in that state they become **indurated**:
irreversibly lost. Collapse precedes fibrosis rather than following it.

## Running it

```python
from alveolar import AlveolarConfig, run_and_record

config = AlveolarConfig(
    total_time_h=1440.0,            # 60 days
    repair_failure_factor=0.10,     # severity of the AT2→AT1 block
    stall_promotion_strength=25.0,  # mesenchyme → epithelium feedback
    aberrant_clearance_rate=0.0005, # apoptosis resistance (lower = more resistant)
)
run_and_record(config, "results/progressive", frame_every_h=24.0)
```

A 60-day run takes about 20 s including rendering.

## The knobs that decide the outcome

| Parameter | Meaning | IPF correlate |
| --- | --- | --- |
| `repair_failure_factor` | how badly AT2→AT1 is blocked | senescent/dysfunctional AT2 |
| `injury_activation_boost` | how hard AT2 are pushed to differentiate | repetitive micro-injury |
| `stall_promotion_strength` | how strongly profibrotic signal keeps cells KRT8+ | the K8-dependent feedback loop |
| `repair_inhibition_strength` | how strongly it blocks the exit to AT1 | IL11, TGF-β signalling |
| `aberrant_clearance_rate` | removal of aberrant cells | apoptosis resistance of senescent cells |
| `aberrant_emt_rate` | conversion to mesenchyme | EMT; the input to the focus model |

Mechanical knobs: `surfactant_production_per_h`, `surfactant_loss_per_h`,
`surface_tension_min/max_mN_m`, `tissue_recoil_Pa`, `induration_time_h`,
`collapse_damage_factor`.

## What the scans showed

With the healthy baseline the lung stays healthy: aberrant fraction ~0.6 %,
AT1 coverage ~82 %, no runaway. That stability is a requirement, not a result —
a model whose healthy state drifts into disease is useless.

Introducing the lesion produces a proportionate response that **does not**
become self-sustaining at baseline feedback. Progression appears only above a
threshold in the feedback loop:

| `stall_promotion_strength` | `aberrant_clearance_rate` | outcome |
| --- | --- | --- |
| 4 | 0.0025 – 0.0003 | stable |
| 12 | 0.0025 – 0.0010 | stable |
| 12 | 0.0003 | **progresses** |
| 25 | any | **progresses** |

So the epithelial point of no return is governed primarily by the strength of
the mesenchyme→epithelium feedback, with apoptosis resistance lowering the
threshold. This is a *separate* bistability from the matrix one in the
fibroblastic focus model: one lives in the epithelium, one in the ECM.

## Caveats

- Two-dimensional. Real alveoli are 3D polyhedra sharing septa with many
  neighbours; this is a section through that structure.
- Alveolar collapse is treated as a discrete open/closed switch with a fixed
  radius reduction, not a mechanical relaxation of the septal network.
- The profibrotic signal is a single lumped field; IL11, TGF-β and the
  macrophage compartment are not separated.
- The numeric thresholds above are model outputs, not measurements. Several
  input rates are order-of-magnitude estimates, so treat the *existence* and
  *structure* of the boundary as the result, not its exact position.


---

# Stage 3: the coupled model

`mesenchyme.py` adds the layer that turns a collapsed alveolus into a
fibroblastic focus, and `render.run_and_record_coupled` draws all three layers
at once.

## The confinement is the mechanism

Fibroblasts do not live in air. In an intact alveolus they are restricted to the
interstitium — the thin space inside the septum. That confinement is why a
healthy lung does not fill in, and it is enforced explicitly: a move into open
air space is rejected.

When an alveolus derecruits, its volume stops being air space and becomes
available. Cells released by EMT at aberrant epithelial segments, plus resident
fibroblasts recruited from neighbouring septa by durotaxis and by chemotaxis up
the profibrotic gradient, migrate in. Only there does packing climb high enough
for nematic order — and therefore for ±1/2 defects — to exist at all.

```python
from alveolar import AlveolarConfig, run_and_record_coupled

config = AlveolarConfig(
    total_time_h=1440.0,
    repair_failure_factor=0.10,
    stall_promotion_strength=25.0,
    aberrant_clearance_rate=0.0005,
)
run_and_record_coupled(config, "results/coupled", frame_every_h=24.0)
```

A 60-day coupled run takes about 40 s, or 56 s with rendering.

## What the coupled run produces

| day | aberrant | indurated | mesenchymal | myofibroblast | packing | E max | septum | defects |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.0 % | 0 % | 260 | 0 | 0.76 | 2 kPa | 9.0 µm | 0 |
| 20 | 1.6 % | 5 % | 453 | 81 | 0.62 | 10 kPa | 9.2 µm | 4 |
| 40 | 5.6 % | 13 % | 883 | 457 | 1.18 | 41 kPa | 12.0 µm | 17 |
| 60 | 10.6 % | 15 % | 1288 | 917 | 1.23 | 45 kPa | 15.5 µm | 23 |

Septal thickening from 9.0 to 15.5 µm is a directly measurable prediction:
it can be read off H&E sections without any new staining.

## Spatial test of the hypothesis

The claim is that foci form *in collapsed alveoli*. Measured at day 60 in one
realisation:

- 52 % of defects lie inside collapsed or indurated alveoli, which occupy only
  21 % of the tissue area — chance would put 4.9 defects there, the model puts
  12 (≈2.4× enrichment);
- 60 % of all mesenchymal cells sit in that same 21 % of area;
- the myofibroblast fraction is 88 % inside collapsed alveoli against 47 %
  outside.

So collapse concentrates the mesenchyme, the activated phenotype and the
topological defects together. This is one realisation with one seed: it shows
the mechanism operates, not that the effect size is reliable. Repeat seeds
before quoting numbers.

## Further caveats for stage 3

- Alveolar collapse changes an alveolus's radius by a fixed factor rather than
  by relaxing the septal network mechanically, so the geometry of a collapsed
  alveolus is only schematic.
- The mesenchyme is confined by a mask derived from distance to the septal
  midline; septal thickening feeds back on that mask, but the septum has no
  independent mechanics.
- Cells are removed from the epithelium on EMT and added to the mesenchyme with
  a myofibroblast phenotype already set; there is no intermediate.
- Defect counts are sensitive to `coarse_grain_um` and
  `min_packing_for_nematic`. Match these to the histology analysis in physical
  units before comparing defect densities between simulation and tissue.


---

# Stage 4: breathing, and a corrected clock

Two changes, one physical and one about honesty.

## Breathing: the tidal volume is conserved, so strain redistributes

The chest wall imposes a fixed tidal volume. Each ventilated region takes a
share of the deformation proportional to its compliance (1/E), and derecruited
alveoli take none at all. So stiff and collapsed regions deform less *because*
they are stiff, and whatever is still healthy must deform more to make up the
difference. Losing alveoli concentrates strain on the survivors — the "baby
lung" effect, and the reason a fibrotic lung is anisotropic rather than
uniformly stiff.

Breathing then enters the model in three places, with different signs:

| Where | Effect | Sign |
| --- | --- | --- |
| Inside stiff/collapsed regions | strain shielding removes the protective stretch signal | **pro**fibrotic |
| On stiff matrix | stretch-activated TGF-β release scales with stiffness | **pro**fibrotic |
| On adjacent healthy tissue | overstrain injures the epithelium, nucleating new lesions | **pro**fibrotic (spatially) |

The first is the counterintuitive one. Cyclic stretch has been reported to
*reduce* fibroblast-to-myofibroblast differentiation, so breathing is
protective at the cell level. A collapsed alveolus does not move, so cells that
migrate into it lose that protection and convert. That is a second positive
feedback on top of the stiffness one, and it explains why the focus grows
specifically where the lung has stopped moving — not merely because there is
room there.

Measured amplification over two years: ventilated tissue goes from 1.00× to
**1.28×** normal tidal strain as 33 % of alveoli are lost.

## The clock was wrong

The earlier runs put 15 % alveolar induration at 60 days. Real IPF takes years:
roughly 47 % of newly diagnosed patients meet a progression endpoint at 12
months. Two separate errors were involved.

**Rates too fast.** Matrix turnover at `0.004 /h` implies a 10-day half-life;
lung collagen turns over on a scale of months. `rate_scale` now multiplies every
intrinsic kinetic rate and divides every intrinsic time, stretching the whole
timescale *without changing a single ratio* — so the bistable structure found
earlier is preserved exactly. The default `rate_scale = 0.08` puts matrix
turnover near 100 days and the disease course near two years. Setting it to 1.0
recovers the fast exploratory behaviour.

**Focus time is not disease time.** The model follows one lesion. An individual
fibroblastic focus may well form in weeks; the disease takes years because it is
the accumulation of many focal events spreading spatially. Labelling the earlier
axis "day 60" and comparing it to clinical progression was a category error.
Overstrain-driven micro-injury of the tissue adjacent to the lesion now supplies
the propagation mechanism, so the slow clock has a cause rather than a fitted
rate.

## Two-year coupled run

| month | aberrant | indurated | mesenchymal | myofibroblast | E max | septum | strain amp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 % | 0 % | 260 | 0 | 2 kPa | 9.0 µm | 1.00× |
| 8 | 12 % | 3 % | 526 | 202 | 8 kPa | 9.7 µm | 1.06× |
| 12 | 17 % | 8 % | 1004 | 670 | 18 kPa | 11.9 µm | 1.19× |
| 24 | 38 % | 33 % | 3836 | 3836 | 45 kPa | 29.3 µm | 1.28× |

New parameters: `tidal_strain`, `breaths_per_min`, `strain_protection_strength`,
`strain_tgfb_gain`, `overstrain_threshold`, `overstrain_injury_gain`,
`rate_scale`.

## Caveats specific to this stage

- Breathing is quasi-static: the tidal strain *amplitude* is modelled and the
  cycle rate enters as a multiplier. Individual breaths are not resolved, which
  is required for a two-year run and is fine for rate-limited processes, but
  would be wrong for anything depending on waveform or frequency.
- Strain is distributed by a local compliance rule, not by solving elasticity.
  There is no shear, no stress at the interface itself, and no septal
  mechanics — so the *pattern* of strain redistribution is right but its
  magnitude near sharp boundaries is not to be trusted.
- `rate_scale = 0.08` was chosen so the disease course lands near two years. It
  is a calibration to a clinical impression, not a fit to data. The alveolar
  loss fraction from the literature is the number that would replace it.
- At month 24 essentially every mesenchymal cell is a myofibroblast, which is
  almost certainly too complete; it suggests the strain-protection term needs a
  floor, or that reversion is missing.


---

# Stage 5: defect tracking — and a negative result that matters

`defect_tracking.py` links defects between snapshots into trajectories, measures
lifetimes, and samples the local tissue state along each track, so that "which
defects become stable, and do they hold the phenotype?" becomes a measurement
rather than an impression.

Applied to the model, the answer is: **none of them are stable, and the reason
invalidates the measurement rather than answering it.**

## What was found

Over one simulated year with daily sampling, 2 629 defect tracks were built.
Ninety-two per cent appeared in a single frame; the longest-lived survived four
days. Sampling every timestep (2 h) instead of daily changed nothing: 93 % still
appeared once, maximum lifetime four hours.

Two candidate explanations were tested and both were rejected:

- **Timescale mismatch.** `rate_scale` had been applied to the biology but not
  to cell mechanics, so orientations randomised fully within one step. Fixed by
  making mobility matrix-dependent — but defects still did not persist.
- **Insufficient pinning.** Matrix immobilization was raised from 25 to 5 000,
  cutting median mobility from 0.54 to 0.0046 (a 200-fold slowdown). Ninety per
  cent of defects were still single-frame.

## The actual cause: the director field is shot-noise limited

Counting cells inside one coarse-graining radius (28 µm) gives a **median of 4**.
For N randomly oriented rods, pure counting noise produces an apparent nematic
order of about 1/sqrt(N):

| N cells per window | spurious \|S\| from noise alone |
| --- | --- |
| 3 | 0.58 |
| 5 | 0.45 |
| 10 | 0.32 |
| 30 | 0.18 |
| 100 | 0.10 |

The order actually measured in gated regions is **S ≈ 0.32**, which is *smaller*
than the 0.5 that four randomly oriented cells would produce by chance. The
nematic order in this model is therefore not distinguishable from counting
noise, and the "defects" detected in it are noise features with no persistent
identity. Nothing about their stability or their phenotype can be concluded.

This is a property of the measurement, not of the biology: the mesenchyme is
confined to a septum a few microns thick, so any window wide enough to average
over cells is also wide enough to cross the septum. Raising the cell count,
narrowing the coarse-graining, or restricting the analysis to filled collapsed
alveoli (where cells are genuinely dense in 2D) are the three ways out.

## Why this matters outside the simulation

The same arithmetic applies to histology. Before interpreting a defect map from
tissue, count how many orientation samples fall inside one coarse-graining
window; below roughly 30, apparent order is dominated by counting noise.

This may already be visible in the tissue results obtained earlier: the collagen
director field, built from a structure tensor over continuous eosin intensity,
has many effectively independent samples per window and showed strong
enrichment against the permutation null; the nuclear director field, built from
a modest number of discrete segmented nuclei, sat at chance. That contrast is
what a shot-noise-limited estimator looks like, and it suggests the nuclear
route needs either denser sampling or a wider window before its defect counts
mean anything.

## Using the tracker

```python
from alveolar import DefectTracker, make_sampler, random_control

tracker = DefectTracker(max_displacement_um=40.0)
for step in range(n_steps):
    sim.step()
    if step % every == 0:
        tracker.update(sim.time_h, sim.mesenchyme.detect_defects(),
                       make_sampler(sim))

summary = tracker.summary(stable_threshold_h=24 * 14)
control = random_control(sim, n_points=400)
```

`summary` splits tracks by charge into stable and transient and reports, for
each group, the mean local myofibroblast fraction, stiffness, tidal strain,
whether the site lies in a collapsed alveolus, and how far the defect drifted.
`random_control` measures the same quantities at random permitted locations, so
any enrichment is stated against a null rather than in absolute terms.

The machinery is correct and validated; it is the model's director field that
cannot currently feed it. Re-run this analysis once cell density inside filled
alveoli is high enough that S exceeds the noise floor for the window in use.

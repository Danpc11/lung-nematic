# simulations / alveolar

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
- Mesenchymal cells released by EMT are counted but not yet placed — that is
  the coupling to the focus model, and is the next stage.
- The numeric thresholds above are model outputs, not measurements. Several
  input rates are order-of-magnitude estimates, so treat the *existence* and
  *structure* of the boundary as the result, not its exact position.

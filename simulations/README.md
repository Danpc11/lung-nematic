# simulations

Two models of pulmonary fibrosis, plus the analysis that joins them. They live
in separate subpackages because each defines a `model.py` and a `render.py`, and
flattening them silently overwrites files.

```
simulations/
├── alveolar/            alveolar architecture, epithelial state machine,
│                        breathing, coupled mesenchyme, defect tracking
├── fibrofocus/          standalone fibroblastic-focus model and its
│                        bistability analysis of the point of no return
├── coupled_analysis.py  joins the epithelial and matrix bistabilities
└── configs/             parameter sets that reproduce specific runs
```

Each subpackage has its own README with the model description, parameter
provenance and caveats. Start there:

- [`alveolar/README.md`](alveolar/README.md) — the main model, built in six
  stages, including two results that are negative and one bug worth knowing
  about.
- [`fibrofocus/README.md`](fibrofocus/README.md) — the standalone focus model
  and the reduced equation that locates the point of no return.

```python
from simulations.alveolar import AlveolarConfig, run_and_record_coupled
from simulations.fibrofocus import FocusConfig, critical_value
from simulations.coupled_analysis import CoupledParameters, loop_interruption_test
```

## Reproducing a run

`run_and_record_coupled` writes a `config.json` beside its outputs recording
every parameter used. That file is a *run artefact* and lands in a gitignored
results directory, so it is not version-controlled by default.

Parameter sets that back a figure or a claim belong in `configs/` instead, which
is committed. To rerun one:

```python
import json
from dataclasses import replace
from simulations.alveolar import AlveolarConfig, run_and_record_coupled

with open("simulations/configs/two_year_progressive.json") as handle:
    config = replace(AlveolarConfig(), **json.load(handle))
run_and_record_coupled(config, "results/two_year", frame_every_h=292.0)
```

---

## Coupling the two analyses — and a retraction

`coupled_analysis.py` joins the epithelial and matrix bistabilities that had
only ever been studied separately. Three variables, reduced from the agent
models: aberrant epithelial fraction `A`, myofibroblast density `M`, matrix
stiffness `E`. Four couplings, each separately switchable:

| symbol | arrow | meaning |
| --- | --- | --- |
| `beta` | A → P → A | epithelial self-promotion |
| `r_activate` | M → E → M | matrix self-promotion |
| `eta` | A → M | EMT: epithelium supplies mesenchyme |
| `lam` | E → P | stretch-activated TGF-β: mesenchyme talks back |

### A prediction of mine was wrong

Earlier I wrote that two separate bistabilities "predict that breaking either
one suffices." **The coupled model says otherwise.** Cutting either
self-promotion loop alone leaves the lesion fibrotic:

| intervention | A | M | E (kPa) | outcome |
| --- | --- | --- | --- | --- |
| intact | 0.88 | 1.00 | 30.0 | fibrotic |
| epithelial loop cut (`beta`=0) | 0.10 | 0.96 | 29.1 | **still fibrotic** |
| matrix loop cut (`r_activate`=0) | 0.88 | 1.00 | 30.0 | **still fibrotic** |
| stretch TGF-β cut (`lam`=0) | 0.68 | 1.00 | 30.0 | **still fibrotic** |
| EMT link cut (`eta`=0) | 0.70 | 0.02 | 2.8 | resolves |
| both loops cut | 0.10 | 0.25 | 10.9 | resolves |
| clearance restored (×20) | 0.69 | 1.00 | 30.0 | still fibrotic |
| matrix turnover restored (×5) | 0.78 | 1.00 | 9.3 | resolves |

The two switches are not in series. Each compartment can sustain the fibrotic
state on its own through the cross-couplings, so disabling one self-promotion
loop simply hands the job to the other. What resolves the lesion is cutting the
**link** between compartments, or restoring matrix turnover.

### Matrix turnover dominates

Scanning epithelial loop strength against turnover, the outcome is set almost
entirely by turnover:

| `beta` | `k_deg` = 0.004 | `k_deg` = 0.012 |
| --- | --- | --- |
| 2 | fibrotic (A 0.27, E 30) | healthy (A 0.13, E 7) |
| 6 | fibrotic (A 0.52, E 30) | healthy (A 0.36, E 13) |
| 12 | fibrotic (A 0.69, E 30) | healthy (A 0.55, E 13) |
| 25 | fibrotic (A 0.83, E 30) | healthy (A 0.74, E 13) |

Note what `A` does in the right-hand column: the aberrant epithelial fraction
stays high (up to 0.74) while the matrix resolves. **Persistent aberrant
epithelium without progressive fibrosis is a stable state of this model** — which
would explain why aberrant basaloid cells can be present without the patient
progressing, and predicts that epithelial abnormality alone is a poor marker of
progression.

### A bug this analysis exposed

Building the reduction gave a myofibroblast a 43-month lifetime. `rate_scale`
had been applied to *every* rate including proliferation, death and migration —
but those are cell-biological processes with their own clock of hours to weeks,
and they do not slow down because the disease is slow. `rate_scale` now covers
only matrix and disease kinetics (`UNSCALED_CELL_RATES` lists the exclusions),
which puts myofibroblast lifetime at 3.4 months.

### Caveats

- In the reduction, EMT is the *only* path from epithelium to mesenchyme, so its
  apparent indispensability is partly structural. The agent model also recruits
  resident fibroblasts by durotaxis and chemotaxis; the reduced model almost
  certainly overstates EMT.
- No bistability was found in the coupled system at these parameters, although
  each subsystem was bistable alone. Whether coupling genuinely destroys the
  bistability or the reduction is too crude to keep it is unresolved, and worth
  checking against the agent model before it is believed.
- Everything spatial is discarded: confinement, collapse providing room, the
  nematic texture. Use this to locate thresholds and identify which loop carries
  them, then confirm with `alveolar`.

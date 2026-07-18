# fibrofocus

Agent-based active-nematic simulation of **fibroblastic focus** formation in
idiopathic pulmonary fibrosis, built to locate the **point of no return**: the
parameter values beyond which a lesion becomes self-sustaining after the
epithelial insult is withdrawn.

Elongated agents (fibroblasts and myofibroblasts) migrate, align nematically,
proliferate and deposit collagen on a substrate whose stiffness evolves in
response. Nematic order and `±1/2` topological defects are **not imposed**:
they emerge once density crosses the isotropic–nematic crossover, as in
confluent fibroblast monolayers.

---
## The mechanism

1. An **ATII lesion** in the centre of the domain lays down a provisional
   matrix, stiffening it above the mechanical TGF-β activation threshold but
   below the myofibroblast threshold. The insult primes without committing.
2. **Durotaxis** drags fibroblasts up the stiffness gradient into the lesion.
3. Cells that experience stiffness above threshold **activate** into
   myofibroblasts, with strong **hysteresis**: reversion is roughly twenty
   times slower than activation (mechanical memory).
4. Myofibroblasts **deposit collagen**, stiffening the matrix further, which
   recruits and activates more cells: a positive feedback loop.
5. **Matrix turnover** opposes it. Whether the focus persists is decided by the
   competition between deposition and turnover, and by how much memory the
   cells carry.

The insult stops at `injury_duration_h`. What happens afterwards is the whole
question.

---
## Two ways to ask the question

**`model.py` + `render.py` — the agent simulation.** Full spatial dynamics,
emergent nematic texture, defects, and a movie. Expensive (~1 min per run).

**`bistability.py` — the reduced model.** The same biology collapsed onto one
equation for lesion stiffness:

```
dE/dt = k_dep · f(E) · (1 − E/E_max) − k_deg · (E − E_healthy)
```

with `f(E)` the steady-state myofibroblast fraction from the same
hysteretic switch. When `dE/dt` has three roots the system is **bistable**:

```
E_healthy   <   E_separatrix   <   E_fibrotic
 stable          UNSTABLE          stable
```

`E_separatrix` **is** the point of no return. This runs in milliseconds, so it
can be bisected and scanned, then checked against the agent model.

---
## Install and use

```bash
pip install numpy scipy pandas matplotlib imageio imageio-ffmpeg
```

Run a scenario and write the movie:

```bash
python -m fibrofocus.cli run --output results/persistent \
    --deposition-rate-kPa-per-h 0.16 --injury-duration-h 96
```

Find the critical value of any parameter:

```bash
python -m fibrofocus.cli critical \
    --parameter deposition-rate-kPa-per-h --low 0.01 --high 0.6
```

Phase diagram over two parameters:

```bash
python -m fibrofocus.cli scan \
    --x deposition_rate_kPa_per_h --x-range 0.02 0.30 40 \
    --y degradation_rate_per_h   --y-range 0.001 0.02 34 \
    --output results/phase.csv
```

Every parameter in `FocusConfig` is exposed, so the appearance of the focus and
the density of defects are both directly controllable.

---
## Which knob does what

**Controls whether a focus forms and persists** (the no-return group):

| Parameter | Meaning | IPF correlate |
| --- | --- | --- |
| `deposition_rate_kPa_per_h` | collagen output per myofibroblast | procollagen synthesis; the target of antifibrotics |
| `degradation_rate_per_h` | matrix turnover | MMP/TIMP balance; resolution capacity |
| `memory_factor`, `deactivation_rate_per_h` | hysteresis depth | mechanical memory; apoptosis resistance of the myofibroblast |
| `injury_duration_h` | insult length | repetitive micro-injury of the alveolar epithelium |
| `injury_provisional_E_kPa` | provisional matrix stiffness | fibrin/fibronectin clot after epithelial damage |
| `E_act_kPa`, `activation_rate_per_h` | switch position and speed | mechanosensitivity (integrin αvβ6, YAP/TAZ, MKL1) |

Cells are drawn, and counted, as ellipses with a real footprint
(`pi/4 * length * width`). Saturation density is derived from that footprint via
`max_packing_fraction`, so the packing fraction is a physical quantity rather
than a free number: with a C2C12-sized cell this reproduces the ~8.2e-3 /um^2
measured for those monolayers. Defect detection is gated on absolute packing,
because a dilute layer has no nematic phase and any winding in it is noise.

**Controls the nematic texture and defect density** (independent of the switch):

| Parameter | Effect |
| --- | --- |
| `rot_diffusion_per_h` | orientational noise; raising it shortens the correlation length and **multiplies defects** |
| `align_rate_per_h` | nematic coupling (∼ Frank constant); raising it smooths the texture and **removes defects** |
| `speed_um_per_h` | activity; sets the active length scale |
| `prolif_rate_per_h`, `carrying_density_per_um2` | density, and therefore whether nematic order exists at all |
| `rod_length_um` / `rod_width_um` | aspect ratio *and* cell footprint; sets the saturation density |
| `max_packing_fraction` | area fraction at which growth stops |

Defect density scales roughly as `1/ℓ²` with `ℓ = √(K/ζ)`, so the ratio of
`align_rate_per_h` to `rot_diffusion_per_h` is the practical dial.

---
## Parameter provenance

Mechanical parameters come from measurements on spindle-shaped cell monolayers
(Blanch-Mercader et al., *Phys. Rev. Lett.* **126**, 028101, 2021): collective
speed ≈ 21.4 µm/h, cell density ≈ 8.2×10⁻³ µm⁻², rod-like flow alignment.

Stiffness thresholds follow lung mechanobiology (Hinz, *Proc. Am. Thorac. Soc.*
**9**, 137, 2012, and work reviewed there): healthy parenchyma 0.2–2 kPa keeps
fibroblasts quiescent; mechanical TGF-β1 activation needs roughly >5 kPa;
myofibroblast phenotype induction and maintenance sits near 16 kPa; established
fibrosis reaches 20–100 kPa. Fibroblasts primed on stiff substrates keep the
phenotype for about two weeks on soft substrates — the basis for
`memory_factor` and the slow `deactivation_rate_per_h`.

Focus geometry targets come from 3D morphometry (Jones et al., *JCI Insight*
**1**, e86375, 2016): discrete, non-interconnected foci, volumes ~1.3×10⁴ to
9.9×10⁷ µm³, 0.9–11.1 per mm³.

---
## Limitations

- Two-dimensional. Real foci are 3D structures sectioned obliquely in histology.
- Collagen enters only through a scalar stiffness field; its own nematic order
  and the reciprocal cell–matrix alignment are not resolved.
- TGF-β is represented implicitly as a lowered activation threshold inside the
  lesion, not as a diffusing species.
- The epithelium is not an explicit phase, so "epithelial displacement" appears
  as lesion growth rather than as a moving interface.
- The reduced model ignores space entirely; use it to locate thresholds, then
  confirm with the agent model.
- Critical values are **model outputs, not measurements**. They are only as good
  as the parameters above, several of which are order-of-magnitude estimates.

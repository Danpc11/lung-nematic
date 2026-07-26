# Small true-3D alveolar fibrosis model

This package is separate from `simulations/alveolar`, which remains the
auditable 2D/2.5D model. Here, all spatial quantities are genuinely
three-dimensional:

- alveolar centres and respiratory radii;
- epithelial cell positions and surface normals;
- fibroblast positions, migration and nematic orientations;
- 3D neighbour searches, steric displacement and alignment;
- thin interstitial shells and the filled volume of a collapsed alveolus.

The default geometry is one central alveolus with six near-touching neighbours.
The 5 µm space between their surfaces represents a deliberately thin
interstitial compartment. Each alveolus starts with 20 AT2 and 12 AT1 cells.
This 1.67 ratio follows the estimate of 67 AT2 per
40 AT1 cells in an average human alveolus, scaled to the smaller 150 µm
simulated alveolus.

AT2 cells are cuboidal and visibly thicker, but human morphometry does not
support describing them as the larger or less numerous population. Mean AT1
volume is about 1,764 µm³ versus 889 µm³ for AT2. More importantly, one thin,
branched AT1 cell covers about 5,098 µm² of air-facing surface versus 183 µm²
for one AT2. The renderer therefore draws AT1 as broad translucent tangential
plates and AT2 as compact filled cells. With the default counts, represented
surface is approximately 94.4% AT1 and 5.6% AT2.

Morphometric sources:

- [Crapo et al., normal human lung morphometry](https://pubmed.ncbi.nlm.nih.gov/7103258/)
- [Stone et al., mammalian cell number and size](https://pubmed.ncbi.nlm.nih.gov/1540387/)
- [Weibel, topology of human AT1 cells](https://www.atsjournals.org/doi/full/10.1164/rccm.201409-1663OE)

## Run

Install the simulation dependencies and execute:

```bash
pip install -e ".[simulation]"

python -m simulations.alveolar3d.demo \
  --output alveolar_3d_output \
  --days 12 \
  --state-every-hours 24 \
  --breathing-frames 6 \
  --fps 8
```

For a slower 30-day visualization:

```bash
python -m simulations.alveolar3d.demo \
  --output alveolar_3d_30d \
  --days 30 \
  --state-every-hours 48 \
  --breathing-frames 6 \
  --rate-scale 0.35 \
  --fps 8
```

For the human chronic-timescale preset (two calendar years, one saved state
per mean month):

```bash
python -m simulations.alveolar3d.demo \
  --preset human-chronic \
  --output alveolar_3d_human_2y \
  --breathing-frames 6 \
  --fps 8
```

`--rate-scale` multiplies every biological kinetic increment while leaving the
displayed clock in real days. It changes the speed, not the topology or spatial
dimensions of the model.

## Temporal calibration

There is no direct human measurement of the elapsed time from one AT2
transition to formation of one fibroblastic focus. Therefore, the
`human-chronic` preset is a transparent disease-scale calibration rather than
a claim that every per-cell rate is known:

- epithelial transitional states emerge over days in experimental repair
  systems, which constrains the order of events but not the human IPF clock;
- longitudinal collagen biomarkers can change over the first three months;
- prospective IPF cohorts evaluate progression at 6 and 12 months;
- pivotal antifibrotic trials use a 52-week lung-function endpoint.

The preset maps the representative 12-day visual trajectory onto 730.5
calendar days (`kinetic_rate_scale = 12 / 730.5`). It normally places visible
alveolar loss and focus consolidation on a months-to-years axis. Keep this
constant when changing `--days`; otherwise separate experiments would no
longer share the same clock.

Evidence used for this calibration:

- [PFBIO prospective IPF cohort](https://pmc.ncbi.nlm.nih.gov/articles/PMC8281632/)
- [PROFILE collagen biomarker study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6624898/)
- [INPULSIS 52-week trials](https://pubmed.ncbi.nlm.nih.gov/24836310/)
- [AT2 transitional-state kinetics](https://pubmed.ncbi.nlm.nih.gov/37768734/)

MP4 export uses `imageio-ffmpeg` when installed. A system FFmpeg executable can
also be selected explicitly:

```bash
export LUNG_NEMATIC_FFMPEG=/path/to/ffmpeg
```

The output directory contains:

- `alveolar_fibrosis_3d.gif`;
- `alveolar_fibrosis_3d.mp4` when FFmpeg is available;
- every PNG frame;
- model and render configuration JSON files;
- `timeseries_3d.tsv`.

## What the respiratory cycle does

The biological integrator stores local tidal-strain amplitude. The renderer
resolves a representative breath for each saved biological state. Compliant
open alveoli expand radially, stiff alveoli expand less, and collapsed or
indurated alveoli do not expand. Conservation of the imposed deformation
redistributes strain over the remaining open units.

## Biological sequence

The central lesion accelerates:

```text
AT2 -> KRT8+ -> AT1
              \-> aberrant -> EMT -> activated mesenchymal cell
```

Loss of AT2 cells lowers surfactant. Collapse opens the former air-space volume
to mesenchymal migration and proliferation. Myofibroblasts deposit matrix,
raise local stiffness, reduce deformation and stabilize the focus.

## Interpretation limits

- Spherical alveolar units are a reduced geometry, not a reconstructed acinus.
- Near-touching units approximate shared septa; their surfaces are not a
  finite-element mesh with a common wall.
- Fields are stored per alveolar volume rather than on a fine voxel grid.
- Rates in `accelerated_3d_demo_config` are accelerated for visualization.
- `human_chronic_3d_config` is calibrated to a population-level clinical
  timescale, not fitted to longitudinal single-cell measurements.
- The model is suitable for mechanism exploration and software validation, not
  individual prognosis.

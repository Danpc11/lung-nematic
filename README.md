# Lung Nematic

[Python](https://img.shields.io/badge/Python-3.10%2B-blue)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Danpc11/lung-nematic/blob/main/lung_nematic_colab.ipynb)

Nematic director fields and candidate topological defects (`+1/2`, `-1/2`) in
lung histology (H&E), with three orientation sources and statistical controls.
It targets fibrotic lung, where the mechanically relevant architecture lives in
the collagen rather than the nuclei.

> **Status:** research tool (TRL ~4). Outputs are exploratory and **not
> clinically validated**. Defects are reported as *candidates*.

## What it does

For each image the pipeline builds a coarse-grained director field, finds
`±1/2` winding singularities that persist across smoothing scales, and (option-
ally) tests them against a permutation null and a colocalization control.

Three orientation sources:

- **nuclear** — from segmented nuclear long axes (cellular regions).
- **collagen** — from the structure tensor of the eosin channel (fiber
  architecture; dense, works where nuclei are sparse).
- **fused** — a confidence-weighted combination of the two, following nuclei
  where cells are dense and collagen where fibers are.

Two statistical controls:

- **permutation null model** — shuffles orientations while holding positions,
  density, mask and detection fixed, so the observed defect count can be
  compared against chance. The nuclear null permutes per-nucleus angles; the
  collagen null permutes per-pixel eosin gradients.
- **colocalization test** — asks whether local order at defect locations
  differs from random tissue (a bootstrap control; local order drops at a
  genuine defect core).

## Install

```bash
pip install "git+https://github.com/Danpc11/lung-nematic.git"
```

or, for development:

```bash
git clone https://github.com/Danpc11/lung-nematic.git
cd lung-nematic
pip install -e .
```

## Usage

### Command line (batch)

```bash
python -m lung_nematic \
    --input path/to/images \
    --output path/to/results \
    --config config/default_config.json
```

Point `--input` at a folder of images (`.jpg`, `.png`, `.tif`, ...). Results are
written per image under `--output`, plus a `summary_metrics.csv`. An optional
`--metadata` CSV (see `metadata_template.csv`) supplies `microns_per_pixel` and
grouping so defect densities come out in mm^-2.

### Colab (no local setup)

Open `notebooks/lung_nematic_colab.ipynb` in Google Colab. The code is hidden;
you upload images (or load them from Drive), pick the field and the analyses
with form controls, run, and download the results.

### Python

```python
from lung_nematic.config import load_config
from lung_nematic.io_utils import read_rgb
from lung_nematic.preprocessing import make_tissue_mask
from lung_nematic.collagen_field import detect_multiscale_collagen_defects
from lung_nematic.null_model import run_collagen_null_model
from lung_nematic.colocalization import run_colocalization

config = load_config("config/default_config.json")
rgb = read_rgb("image.jpg")
mask, hed = make_tissue_mask(rgb)          # eosin channel is hed[:, :, 1]

# collagen defects
defects, fields, _ = detect_multiscale_collagen_defects(hed[:, :, 1], mask, config)

# is the defect count above chance?
null = run_collagen_null_model(hed[:, :, 1], mask, config, n_permutations=199)

# do defects sit in structured regions?
rep = float(config.sigmas_px[len(config.sigmas_px) // 2])
coloc = run_colocalization(defects, fields[rep], mask, config)
```

## Method

Each nucleus (or eosin pixel) contributes a headless orientation. These are
accumulated into a nematic tensor `Q = S (cos 2t, sin 2t)`, smoothed at several
scales `sigmas_px`, and reduced to a director angle `theta` and a local order
`S`. Candidate defects are grid plaquettes whose director winds by `±pi`
(`±1/2` charge); a candidate is kept only if it persists across at least
`min_scales_for_persistence` scales and lies inside dense tissue away from the
edge. The collagen field uses the structure tensor of the eosin channel; the
fiber direction is the dominant gradient rotated by 90 degrees, and coherence
plays the role of local order.

## Package layout

```text
lung_nematic/
├── config.py            AnalysisConfig dataclass, JSON load/save
├── io_utils.py          image discovery, metadata, RGB reading
├── preprocessing.py     tissue mask + HED stain separation
├── segmentation.py      nuclear segmentation and orientation
├── nematic.py           nuclear nematic tensor field
├── collagen_field.py    structure-tensor collagen field
├── fused_field.py       confidence-weighted nuclear + collagen field
├── defects.py           winding detection + multi-scale persistence
├── null_model.py        permutation null (nuclear and collagen)
├── colocalization.py    defect vs local-order bootstrap test
├── metrics.py           per-image summary metrics
├── visualization.py     overlays and diagnostic panels
├── pipeline.py          single-image pipeline
└── batch.py             folder-level batch driver
```

## Outputs

Per image: a tissue mask, nuclear segmentation, a director-field overlay with
marked candidate defects, per-nucleus and per-defect CSVs, a metrics summary
(JSON + CSV), and, when enabled, null-model and colocalization histograms. The
batch driver also writes a combined `summary_metrics.csv` and a per-group
aggregate.

## Limitations

- Candidate defects only; not clinically validated.
- The eosin channel also stains cytoplasm and red cells, not only collagen, so
  the density gate matters for the collagen field.
- The raw defect count can be a low-power statistic; interpret it together with
  the null model and colocalization, not on its own.
- Without `microns_per_pixel`, defect densities in mm^-2 are unavailable.

## License

MIT.

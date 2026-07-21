# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [0.2.0]

Adaptive-radius defect detection, cell and nucleus morphometry, and a set of
correctness fixes to the analysis and simulation code. New tabular outputs are
written as TSV.

### Added

- `lung_nematic/defects_adaptive.py`: topological defect detection with a
  locally adaptive integration radius. The winding is integrated on a ring whose
  radius is read per pixel from a radius map, so the loop encloses a comparable
  number of cells in dense epithelium and sparse fibroblast stroma alike.
  Includes `adaptive_null_model`, which shuffles orientations while preserving
  the density field and the radius map to test the defect count against chance,
  and `defect_order_context`, which checks per image whether the surviving
  defects sit on low-order domain walls.
- `lung_nematic/adaptive_radius.py`: estimates a per-pixel integration radius
  from local nuclear spacing (or, alternatively, from orientation coherence),
  sized to enclose a set number of local cells.
- `lung_nematic/morphometry.py`: nucleus and cell size quantification. Nuclei
  are segmented directly; whole cells are estimated by watershed expansion from
  the nuclei in histology (a territory estimate) and segmented directly from the
  coverage texture in phase contrast. Per-object tables are written as TSV, in
  pixels and, when a scale is given, microns.
- `lung_nematic/phase_contrast.py`: director fields, defect detection and a
  stiffness-order calibration for NHLF cells on defined-stiffness substrates,
  reusing the collagen structure-tensor engine.
- `lung_nematic/defect_features.py`, `defect_classifier.py`, `labeling.py`: an
  interactive real/uncertain/artefact labelling widget and a grouped-validation
  classifier over per-candidate features.
- `simulations/pharmacology.py`: retrospective drug controls that score the
  reduced model against the clinical record, weighting reproduction of the known
  clinical failures (LOXL2, single-target alpha-v-beta-6) most heavily.
- `simulations/nematic_resolution.py`: adaptive smoothing window with a
  per-window counting-noise null, and the derivation of the `R_min` resolution
  floor set by cell size.
- `fibrofocus_colab.ipynb`: the focus model front-end (separatrix, phase
  diagram, resolution-aware defect analysis).
- Interactive field-calibration and adaptive-defect cells in
  `lung_nematic_colab.ipynb`, styled after OrientationJ, that persist their
  results (TSV, parameter JSON, radius map, figure) into the download bundle.
- CI now runs Ruff and validates every notebook with nbformat.

### Fixed

- **Adaptive detector reported defects in disordered fields.** The order around
  the integration ring was computed but never used to accept or reject a
  candidate, so a chance `±1/2` winding in noise was reported as a defect
  (about five per image at order `S ~ 0.02`). A minimum-ring-order gate now
  requires genuine order around the loop, and the shuffled null model quantifies
  significance. Random fields now yield no defects.
- **Regions far from nuclei were assigned the minimum cell size.** In
  `cell_size_from_nuclei` the smoothed count collapsed to zero away from nuclei,
  clamping the vast majority of the tissue to the minimum. Such regions now
  inherit the nearest valid estimate. The method is documented as measuring
  nuclear spacing, not morphological cell size.
- **Periodic boundary mismatch in the focus simulation.** Cells wrapped on a
  periodic domain but the field smoothing used the default reflect mode, which
  double-counted edge cells and manufactured order in the corners (tens of
  percent false positives against a 5 percent threshold). All field smoothing in
  `FocusSimulation` and in `simulations/nematic_resolution.py` now uses
  `mode="wrap"`, with `boundary_mode` exposed as a parameter.
- **Integer defect counts were inflated.** Every ring enclosing an integer
  defect registered it, so a single `+1` produced two dozen detections.
  Detections are now clustered to their centroid, and a `n_ring_detections`
  column records how many were collapsed.
- **Odd-sized domains broke MP4 encoding** in the focus renderer. Frames are
  trimmed to even dimensions before H.264 encoding.
- **The fibrofocus notebook ran the simulation twice.** `run_and_record` now
  returns the final simulation object, so the defect-analysis cell reuses it
  instead of rebuilding and re-stepping.
- **The parallel null model lost its regression test.** The test compared
  `n_jobs=1` against itself; it now compares serial against parallel for both
  the nuclear and collagen routes and asserts identical results.
- Several unused imports and a mid-file import flagged by Ruff.

### Changed

- New tabular outputs (morphometry, the adaptive-defect notebook cell) are
  written as TSV. The older pipeline, batch, labelling and simulation exports
  still emit CSV; migrating them is tracked for a later release.
- Default director-field scale for histology moved toward the OrientationJ
  regime (integration `sigma ~ 20`, grid `~ 18`), which resolves the domain
  structure a large window had averaged away.
- Retrospective drug controls report the discriminating clinical-failure score
  rather than overall agreement, since agreement with targets that worked
  proves little.

## [0.1.0]

Initial release.

### Added

- Histology analysis pipeline: tissue masking and HED stain separation, nuclear
  segmentation and orientation, nuclear and collagen nematic fields, a fused
  field, half-integer and opt-in integer defect detection with multi-scale
  persistence, a permutation null model, a core/annulus colocalization test, and
  per-image and per-group summary metrics.
- Command-line batch driver and a single-image engine (`analyze_image`).
- Mechanism-based simulations: the alveolar model (Voronoi architecture,
  epithelial state machine, breathing, confined mesenchyme, defect tracking) and
  the standalone fibroblastic-focus model with its reduced bistable equation.
- Colab front-ends for histology analysis and the alveolar simulation.

[Unreleased]: https://github.com/Danpc11/lung-nematic/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Danpc11/lung-nematic/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Danpc11/lung-nematic/releases/tag/v0.1.0

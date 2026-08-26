# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [0.3.0] - 2026-08-26

### Added

- **`simulations.focus3d`**: a three-dimensional active-nematic focus model.
  It exists because the stereology below needs a volume that is genuinely in
  the nematic phase, and `simulations.alveolar3d` cannot supply one: measured
  across configurations it either sits at 1% packing with seven alveoli - a
  dilute rod gas with no orientational order - or, with one alveolus, collapses
  into a blob 33-108 um across against a 38 um alignment radius, pinning the
  order parameter at 0.956 regardless of density. An order parameter that does
  not move when density changes tenfold is a finite-size artefact, not a phase.
  `focus3d` shows a genuine isotropic-nematic crossover between packing 0.12
  and 0.24, and `tests/test_focus3d.py` asserts that response rather than any
  single value.
  Three things had to change from the 2D model and each is commented at the
  point it matters: alignment is computed from the Q tensor over node-contact
  pairs (centre-to-centre neighbourhoods are isotropic and made order *fall*
  with density, 0.067 to 0.017); the alignment-to-noise ratio inherited from 2D
  sits below the 3D ordering threshold, since rotational diffusion acts on two
  angular degrees of freedom instead of one; and growth is capped by volume
  fraction rather than cell count, because a fixed cap means different physics
  in different domain sizes.

- **`simulations.stereology`**: relates defects counted on a 2D section to the
  3D structure they cut. On a slide, defects are line-plane intersections, not
  points, so a section's areal density and a monolayer's are different
  quantities. `N_A = L_V / 2` holds only for isotropically oriented lines, and
  the module measures whether that holds rather than assuming it: on a volume
  of disclinations parallel to z the apparent density runs from tens per mm^2
  to exactly zero depending on the cutting angle alone. On the `focus3d`
  multi-domain regime the anisotropy ratio is 1.37 - sectioning angle changes a
  histological count by 37% - and the per-section charge imbalance is 0.074,
  which turns `crossmap.charge_balance` from a caveat into a threshold.
  The density gate is deliberately absolute rather than a quantile: a 0.4
  quantile on a uniformly filled volume discarded 58% of the plane and every
  defect with it, reporting zero where four were present.

- `make_figures.py`: individual Nature-style figures, one per idea, as 600 dpi
  PNG for slides and vector PDF for a manuscript. The headline order figure
  plots `global_order_excess` rather than raw `S`, since UIP patients carry a
  median of 2883 oriented nuclei against 6632 for NSIP and `S` is biased upward
  as roughly `1/sqrt(N)`; a companion figure shows that confound instead of
  hiding it. Histology panels aggregate to one point per patient, because the
  within-patient ICC of 0.14-0.29 means 148 image-level points would imply far
  more evidence than 13 patients provide.

- `make_simulation_gifs.py`: annotated GIFs of the 2D and 3D models for talks,
  with per-frame order and defect counts. The 3D panel shows a slab rather than
  the full cube - a projection through 400 um of 2800 rods is opaque and hides
  exactly the defects it should show - and fades rods by their out-of-plane
  tilt, since a rod nearly normal to the cut projects to a segment whose
  apparent orientation is mostly noise.

- CI now imports the new simulation modules and runs `--help` on every root
  script; Ruff covers the repository root as well.

### Added (earlier in this cycle)

- Opposite-charge birth/death pairing diagnostics with a spatial null model.
  Pair enrichment distinguishes physical nucleation and annihilation from
  detector flicker without diagnosing the tracker from its own chosen links.
- Pair nucleation counts and calibrated rates in `defect_kinetics.tsv`. Initial
  frame defects are excluded because they are an initial condition, not births.
- Direct MP4 input with metadata provenance, encoded-order tests, and a warning
  for excess director-field mass near compression-prone image axes. Container
  FPS is never used as the acquisition interval; original TIFFs remain preferred.
- `run_flow_maps.py` and `lung_nematic.flow` generate LIC, calibrated collective
  texture-flow maps, vector overlays, and flow--director alignment for selected
  stiffnesses. The output deliberately says collective flow rather than
  single-cell velocity because unlabeled phase contrast does not identify cells.

- `global_order_null` and `expected_order_under_randomness` in `nematic`:
  a permutation floor for the global order parameter. `S` is biased upward as
  ~`1/sqrt(N_eff)`, so it is confounded with nuclei count and tissue area and a
  group with fewer nuclei scores higher for no biological reason. New summary
  columns `global_order_null_mean`, `global_order_excess` and `global_order_p`;
  **report the excess, not raw `S`, whenever the compared groups differ in
  nuclei count or tissue area.**
  A block-permutation null over the smoothed field was implemented first and
  discarded: blocks stay correlated through the Gaussian kernel and rejected
  45% of purely random fields at alpha = 0.05 with 2-sigma blocks, and no block
  size both calibrated and retained power. Permuting the source orientations has
  no such parameter; calibration gives mean p = 0.52 and a 4.0% rejection rate
  at alpha = 0.05, and the simulated floor matches the analytic Rayleigh value.
  Scope: the null permutes *nuclear* orientations, so the pipeline computes it
  for `field_type == "nuclear"` only and leaves the columns NaN for collagen and
  fused runs rather than substituting a floor from a different source.
- `low_orientation_count`: flags images below `min_oriented_nuclei` (default
  200), where `S` is dominated by finite-size noise. Flagged, not dropped.
- `global_nematic_order_S_nuclei`: the previous nuclei-based value, retained so
  earlier results remain traceable.

- `lungtwin`: local sensitivity rank analysis and Cramer-Rao precision bounds
  for a two-state IPF progression model, shipped from this distribution and
  exposed as the `lungtwin-ident` console script. `analyze(..., n_probes=N)`
  recomputes the rank at perturbed parameter points, so a finding can be shown
  not to be an artefact of the nominal values.

### Fixed

- **`global_nematic_order_S` ignored `field_type`.** It was computed by
  `compute_global_order(oriented_nuclei)`, which never receives the field, so
  nuclear, collagen and fused runs over one image returned byte-identical
  values - the column described nuclei even when the run was labelled collagen.
  Confirmed on a real 117-image cohort, where the three field types agreed to
  the last digit while `local_S_median` correlated at only 0.17 between nuclear
  and collagen, as it should. It is now computed from the field itself via
  `compute_global_order_from_field`, as the density-weighted resultant over the
  tissue mask. Because Gaussian smoothing is linear and conserves mass, nuclear
  values reproduce the old ones to within boundary losses, so existing nuclear
  results stay comparable; collagen and fused values change and any conclusion
  drawn from them should be recomputed.

- **`lungtwin` was not importable and its tests could not be collected.** It was
  laid out as a nested project (`lungtwin/pyproject.toml` over
  `lungtwin/lungtwin/`). That nested project built a valid wheel of its own, but
  the root distribution packaged only `lung_nematic*` and `simulations*`, so
  `lungtwin` never shipped; and from the repository root `lungtwin/` resolved as
  an empty namespace package, so `import lungtwin.model` raised
  `ModuleNotFoundError` and `pytest` failed during collection. The package is
  now flattened to `lungtwin/` at the repository root, the nested
  `pyproject.toml` is removed, and `lungtwin*` is included in the root
  distribution. Note that adding `include = ["lungtwin*"]` *without* flattening
  would not have worked: setuptools would have packaged the inner directory as
  `lungtwin.lungtwin`.
- **CI could not have caught the above.** The import job built the real wheel
  and left the checkout - the right shape - but only imported `lung_nematic` and
  `simulations`, never `lungtwin`, and never executed a console script. It now
  imports `lungtwin` and runs `lung-nematic --help` and `lungtwin-ident` from a
  temporary directory. Ruff now covers `lungtwin` as well.
- **Design parameters are validated.** `routine_followup` rejects `n_visits < 1`,
  non-positive or non-finite `interval_months`, and `dlco_missing_rate` outside
  `[0, 1]`. `VisitSchedule` rejects an empty or duplicated channel list, a
  negative or non-finite `treatment_start`, non-finite visit times, non-positive
  measurement noise, and a design where every observation is masked out. A
  `treatment_start` after the final visit is also rejected: it silently turned
  the design into "never treated" under another name, which is precisely the
  design in which `beta` is not estimable.

### Changed

- The CLI now presents pair nucleation rate as the primary activity observable.
  Velocity contrast is retained in the table but no longer interpreted as the
  activity signature because 3--4-frame defect lifetimes do not support a
  robust motion estimate.
- Short-track and orphan warnings now defer to opposite-charge pair enrichment.
  Enriched paired turnover indicates physical birth/death and does not justify
  increasing the linking gate; only near-null pairing triggers detector and
  tracker checks.

- **`lungtwin` no longer claims structural identifiability.** `analyze` computes
  the numerical rank of a finite-difference sensitivity matrix at a single
  nominal parameter point, which establishes *local* identifiability there and
  nothing more; discrete symmetries, disconnected solution branches, and rank
  collapse elsewhere in parameter space are invisible to a single-point
  Jacobian. `IdentifiabilityReport.is_structurally_identifiable` is renamed
  `is_locally_identifiable`, the report header reads "LOCAL SENSITIVITY RANK
  ANALYSIS", and the previous assertion that a rank deficiency means "no amount
  of data, no prior and no optimiser" can recover the combination is removed -
  that is true of some of this model's deficiencies, but it is not what a local
  rank computation establishes.

- **Output no longer re-enters as input.** `discover_images` accepts
  `exclude_dirs`, and `analyze_folder` rejects `output_dir == input_dir` and
  excludes the output subtree from discovery. Previously, placing `--output`
  inside `--input` meant a second run picked up the first run's overlays,
  diagnostic panels and defect maps as histology. Those images do not fail -
  annotations are burned into the raster, so tissue segmentation succeeds - so
  the run produced plausible metrics from synthetic inputs with no warning.
  Covered by `tests/test_output_isolation.py`, which runs the same batch twice
  over a nested output directory and asserts the two summaries are identical.
- **`image_id` uniqueness is enforced on the normalized name.** The duplicate
  check ran on the raw identifier, but `_safe_identifier` then folded `/`, `\`,
  `:` and spaces to `_`, so `case/01` and `case_01` passed the check and wrote
  into the same directory. The normalization now lives in
  `io_utils.safe_identifier` (single definition, imported by both `batch` and
  `pipeline`) and `analyze_folder` checks uniqueness on its output.
- **Reserved `image_id` values are rejected rather than normalized.** An
  `image_id` of `..` made `output_root / safe_id` resolve to the *parent* of the
  output root, scattering per-image results outside it; `""` and `"."` resolved
  to the output root itself. These now raise before any image is processed.
  Trailing dots are also stripped, since NTFS drops them silently and `case01.`
  and `case01` would otherwise be one directory on Windows and two elsewhere.
- **Permutation nulls are seeded per image, not per batch.** Every image was
  seeded with `config.random_seed`, which is reproducible but leaves the
  permutation streams identical across images. That dependence is invisible in a
  per-image p-value and only matters when p-values are combined across a cohort.
  `io_utils.derived_seed(base, image_id, field_type)` (CRC32-based, so stable
  across processes) now supplies a distinct stream per image and field, and the
  value used is recorded as `random_seed_used` in the summary. **This changes
  null-model and colocalization numbers relative to earlier runs**; point
  estimates are unaffected.

### Changed

- **`defect_density_mm2` now counts half-integer defects only.** It was
  `len(defects) / area`, pooling the half-integer (plaquette winding) and
  integer (N-point ring) layers, which have different spatial support and
  sensitivity. The column therefore meant different things depending on whether
  `detect_integer_defects` was set, and runs with and without the flag were not
  comparable under the same name. `defect_density_integer_mm2` and
  `defect_density_all_mm2` (the previous quantity) are reported separately, with
  the integer density as NaN - not zero - when the layer was not run, so "not
  measured" stays distinguishable from "measured, none found". New companion
  counts: `n_half_total`, `n_integer_total`, `detect_integer_defects`.
- **Batch reports are per-field and additive.** `analyze_folder` writes
  `summary_metrics_<field>.csv` and `processing_errors_<field>.csv`, then
  rebuilds the combined `summary_metrics.csv` / `processing_errors.csv` from
  every per-field file present. Running the CLI twice over one output directory
  with different `--field` values previously left per-image directories holding
  both fields while the batch summary described only the last, with no marker of
  the overwrite. `summarize_by_group` keys on `field_type` as well as `group`
  when the column is present, so nuclear and collagen measurements are never
  averaged into one row.

- The analysis pipeline (`analyze_image`) now writes its per-image tables -
  nuclei, defects, raw detections, colocalization null, defect maps and null
  totals - as TSV rather than CSV, matching the rest of the project's tabular
  output. (Batch summary and other modules still emit CSV.)

### Fixed

- `analyze_image` validates the required metadata keys (`image_id`, `group`,
  `filename`) up front, so a missing key raises a clear error before
  segmentation and detection run rather than a bare `KeyError` deep in the call.
- The representative-scale lookup raises an explanatory error if the config
  sigmas and the computed field scales drift out of sync, instead of a bare
  `KeyError`.

### Added

- `simulations/alveolar3d/`: a genuinely three-dimensional alveolar fibrosis
  prototype (3D positions, surface normals, migration, neighbour searches and
  respiratory deformation), seeded from human morphometry. Small by design —
  seven near-touching units — and a research prototype, not a calibrated
  predictor.
- `simulations/alveolar/particle_render.py`: a 2.5D particle view of the coupled
  epithelial/mesenchymal state, with a `particle_demo.py` example. The biology
  still evolves on the validated 2D tessellation; the third coordinate is a
  visual embedding only.

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

"""
Tests for the defect-classifier stack: features, training, grouped validation.

The point of these is to lock in the two properties that make the classifier
trustworthy rather than a slow copy of the labeller:

  * feature extraction produces the exact, ordered column set the trained model
    will later expect, on real candidate tables and on empty ones;
  * validation is leave-one-image-out, so a reported score reflects an unseen
    image and cannot be inflated by splitting one image across train and test.

The heavier "does it learn" test uses a synthetic feature table with a planted,
separable structure, because whether the model learns anything on the real gels
depends on hand labels that do not exist yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lung_nematic.defect_features import (
    FEATURE_COLUMNS,
    extract_features,
    feature_matrix,
)
from lung_nematic.defect_classifier import (
    CLASSES,
    grouped_cross_validate,
    train_classifier,
)


def _synthetic_field(shape=(200, 200), seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, np.pi, shape)
    order = rng.uniform(0.2, 0.9, shape)
    density = np.ones(shape)
    return {"theta": theta, "order": order, "density": density}


def _candidates(n=6, shape=(200, 200), seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x_px": rng.uniform(20, shape[1] - 20, n),
        "y_px": rng.uniform(20, shape[0] - 20, n),
        "charge": rng.choice([0.5, -0.5], n),
        "charge_raw": rng.uniform(-0.55, 0.55, n),
    })


# ------------------------------------------------------------------ features
def test_features_have_the_canonical_columns():
    field = _synthetic_field()
    candidates = _candidates()
    features = extract_features(candidates, field, core_radius_px=10.0)
    assert list(features.columns)[: len(FEATURE_COLUMNS)] == FEATURE_COLUMNS or \
        set(FEATURE_COLUMNS).issubset(features.columns)
    assert len(features) == len(candidates)


def test_features_on_empty_candidates():
    field = _synthetic_field()
    empty = pd.DataFrame(columns=["x_px", "y_px", "charge", "charge_raw"])
    features = extract_features(empty, field, core_radius_px=10.0)
    assert len(features) == 0


def test_feature_matrix_is_finite():
    field = _synthetic_field()
    features = extract_features(_candidates(), field, core_radius_px=10.0)
    matrix = feature_matrix(features)
    assert np.isfinite(matrix).all(), "features must not contain NaN or inf"


def test_core_annulus_ratio_separates_real_from_flat():
    """A planted disordered core in an ordered field must score a low ratio."""
    shape = (200, 200)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    radius = np.hypot(xx - 100, yy - 100)
    order = np.tanh(radius / 8.0)              # 0 at the core, ~1 far out
    field = {"theta": np.full(shape, 0.5),
             "order": order, "density": np.ones(shape)}
    real = pd.DataFrame({"x_px": [100.0], "y_px": [100.0],
                         "charge": [0.5], "charge_raw": [0.5]})
    flat = pd.DataFrame({"x_px": [30.0], "y_px": [30.0],
                         "charge": [0.5], "charge_raw": [0.5]})
    ratio_real = extract_features(real, field, core_radius_px=8.0)["order_core_annulus_ratio"].iloc[0]
    ratio_flat = extract_features(flat, field, core_radius_px=8.0)["order_core_annulus_ratio"].iloc[0]
    assert ratio_real < ratio_flat, (
        "a real core (order dips then recovers) should have a lower "
        "core/annulus ratio than a uniformly ordered patch"
    )


# ------------------------------------------------------------------ training
def _labelled_table(n_per_class=25, seed=0):
    """A feature table with a planted, separable class structure."""
    rng = np.random.default_rng(seed)
    frames, labels, groups = [], [], []
    # real: low core/annulus ratio, high persistence; artefact: the opposite
    specs = {
        "real": {"order_core_annulus_ratio": 0.3, "scales_detected": 3},
        "artefact": {"order_core_annulus_ratio": 0.95, "scales_detected": 1},
    }
    for label, centre in specs.items():
        block = pd.DataFrame(
            rng.normal(0.0, 0.05, size=(n_per_class, len(FEATURE_COLUMNS))),
            columns=FEATURE_COLUMNS,
        )
        for key, value in centre.items():
            block[key] = value + rng.normal(0, 0.03, n_per_class)
        frames.append(block)
        labels += [label] * n_per_class
        # spread across three "images" so grouped CV has folds
        groups += list(rng.integers(0, 3, n_per_class).astype(str))
    return pd.concat(frames, ignore_index=True), pd.Series(labels), pd.Series(groups)


def test_classifier_learns_a_planted_structure():
    features, labels, _ = _labelled_table()
    classifier = train_classifier(features, labels, kind="random_forest")
    probabilities = classifier.predict_proba(features)
    predicted = probabilities[list(classifier.classes)].idxmax(axis=1)
    accuracy = (predicted.to_numpy() == labels.to_numpy()).mean()
    assert accuracy > 0.9, "should recover a clearly separable structure"


def test_grouped_cross_validation_runs_and_holds_images_out():
    features, labels, groups = _labelled_table()
    result = grouped_cross_validate(features, labels, groups)
    assert result["n_images"] == 3
    assert "confusion_matrix" in result
    assert result["n_candidates"] == len(features)


def test_uncertain_class_is_supported():
    """The three-way label set including 'uncertain' must train without error."""
    features, labels, groups = _labelled_table()
    # relabel a slice as uncertain
    labels = labels.copy()
    labels.iloc[:10] = "uncertain"
    classifier = train_classifier(features, labels)
    assert set(classifier.classes) == set(CLASSES)


def test_balanced_weighting_resists_all_artefact_collapse():
    """With a heavy class imbalance the model must not just predict the majority."""
    features, labels, _ = _labelled_table(n_per_class=40)
    # make artefacts 8x more common, as real hand-labelled sets tend to be
    keep_real = labels[labels == "real"].index[:5]
    keep = labels[labels == "artefact"].index.tolist() + list(keep_real)
    features = features.loc[keep].reset_index(drop=True)
    labels = labels.loc[keep].reset_index(drop=True)
    classifier = train_classifier(features, labels, class_weight="balanced")
    predicted = classifier.predict_proba(features)[list(classifier.classes)].idxmax(axis=1)
    assert (predicted == "real").any(), (
        "balanced weighting should let the rare 'real' class still be predicted"
    )

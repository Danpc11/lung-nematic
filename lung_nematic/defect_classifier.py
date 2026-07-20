"""
A classifier that learns which defect candidates you would keep.

The detector proposes candidates by fixed rules. This learns, from your labels,
which of those candidates are real, which are artefacts, and which are genuinely
ambiguous - the three classes you chose. Once trained it plugs into the pipeline
as an extra filter: each candidate gets a probability, and you set the threshold
instead of accepting everything that passed the rules.

Two design decisions keep it honest on a small dataset:

* **The model is small and interpretable.** A handful of labelled images yields
  at most a few hundred candidates, far too few for a deep network, which would
  memorise the images. A random forest (or logistic regression) over the
  precomputed features learns from dozens of examples and, more usefully, tells
  you *which feature* separates the classes - a result in itself.

* **Validation is grouped by image, never by candidate.** Two candidates from
  the same frame share lighting, stain, cell line, and your labelling mood that
  day. Splitting them across train and test leaks all of that and reports an
  accuracy you will never see on a new image. Every split here keeps whole
  images together.

The "uncertain" class is not forced into a hard boundary. At inference the model
returns the full probability over the three classes, so a candidate the model is
unsure about surfaces as low confidence rather than a confident wrong call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .defect_features import FEATURE_COLUMNS, feature_matrix

# label vocabulary; order fixed so saved models and reports agree
CLASSES = ("artefact", "uncertain", "real")
CLASS_TO_INT = {name: index for index, name in enumerate(CLASSES)}


@dataclass
class DefectClassifier:
    """A trained model plus everything needed to apply and interpret it."""

    model: object
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    classes: tuple[str, ...] = CLASSES
    kind: str = "random_forest"
    metadata: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- predict
    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """Probability of each class for every candidate."""
        matrix = feature_matrix(features)
        probabilities = self.model.predict_proba(matrix)
        # align to the canonical class order regardless of sklearn's ordering
        columns = {}
        for position, class_index in enumerate(self.model.classes_):
            columns[self.classes[int(class_index)]] = probabilities[:, position]
        result = pd.DataFrame(columns).reindex(columns=list(self.classes))
        result = result.fillna(0.0)
        result.index = features.index
        return result

    def predict(self, features: pd.DataFrame, real_threshold: float = 0.5
                ) -> pd.DataFrame:
        """Label each candidate and attach the class probabilities.

        A candidate is called ``real`` only if its ``real`` probability clears
        ``real_threshold``; otherwise it takes its most probable class. Raising
        the threshold trades recall for precision, and is the knob you tune per
        study.
        """
        proba = self.predict_proba(features)
        called = proba.idxmax(axis=1)
        called[proba["real"] >= real_threshold] = "real"
        out = features.copy()
        for name in self.classes:
            out[f"p_{name}"] = proba[name].to_numpy()
        out["predicted_class"] = called.to_numpy()
        return out

    # --------------------------------------------------------- interpretation
    def feature_importance(self) -> pd.DataFrame:
        """What the model actually keys on. This is a result, not diagnostics."""
        if hasattr(self.model, "feature_importances_"):
            values = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            values = np.abs(self.model.coef_).mean(axis=0)
        else:
            return pd.DataFrame(columns=["feature", "importance"])
        table = pd.DataFrame(
            {"feature": self.feature_names, "importance": values}
        )
        return table.sort_values("importance", ascending=False).reset_index(drop=True)

    # ---------------------------------------------------------------- persist
    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        joblib.dump(self.model, path.with_suffix(".joblib"))
        path.with_suffix(".json").write_text(json.dumps({
            "feature_names": self.feature_names,
            "classes": list(self.classes),
            "kind": self.kind,
            "metadata": self.metadata,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DefectClassifier":
        import joblib

        path = Path(path)
        model = joblib.load(path.with_suffix(".joblib"))
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(model=model, feature_names=meta["feature_names"],
                   classes=tuple(meta["classes"]), kind=meta["kind"],
                   metadata=meta.get("metadata", {}))


def train_classifier(
    features: pd.DataFrame,
    labels: pd.Series,
    kind: str = "random_forest",
    class_weight: str | dict | None = "balanced",
    seed: int = 0,
) -> DefectClassifier:
    """Fit a classifier on labelled candidates.

    ``labels`` holds one of ``artefact`` / ``uncertain`` / ``real`` per row.
    ``class_weight="balanced"`` matters here: hand-labelled sets are almost
    always mostly artefacts, and without reweighting the model learns to call
    everything an artefact and reports a high accuracy for doing nothing.
    """
    matrix = feature_matrix(features)
    y = labels.map(CLASS_TO_INT).to_numpy()

    if kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight=class_weight, random_state=seed, n_jobs=-1,
        )
    elif kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight=class_weight, max_iter=2000,
                multi_class="multinomial", random_state=seed,
            ),
        )
    else:
        raise ValueError(f"unknown classifier kind: {kind}")

    model.fit(matrix, y)
    return DefectClassifier(
        model=model, kind=kind,
        metadata={"n_train": int(len(y)),
                  "class_counts": labels.value_counts().to_dict()},
    )


def grouped_cross_validate(
    features: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
    kind: str = "random_forest",
    seed: int = 0,
) -> dict:
    """Leave-one-image-out validation, the only kind that is not self-deceiving.

    Candidates from the same image are never split across train and test, so the
    reported numbers reflect performance on an *unseen image* - which is what
    you will actually run the model on.
    """
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import LeaveOneGroupOut

    matrix = feature_matrix(features)
    y = labels.map(CLASS_TO_INT).to_numpy()
    group_values = groups.to_numpy()

    splitter = LeaveOneGroupOut()
    predictions = np.empty_like(y)
    fold_scores = []

    for train_index, test_index in splitter.split(matrix, y, group_values):
        if len(np.unique(y[train_index])) < 2:
            predictions[test_index] = y[train_index][0]
            continue
        fold = train_classifier(
            features.iloc[train_index], labels.iloc[train_index],
            kind=kind, seed=seed,
        )
        fold_pred = fold.model.predict(matrix[test_index])
        predictions[test_index] = fold_pred
        held_out_image = groups.iloc[test_index].iloc[0]
        fold_scores.append({
            "held_out_image": held_out_image,
            "n": int(len(test_index)),
            "accuracy": float((fold_pred == y[test_index]).mean()),
        })

    present = sorted(np.unique(y))
    report = classification_report(
        y, predictions,
        labels=present,
        target_names=[CLASSES[i] for i in present],
        output_dict=True, zero_division=0,
    )
    matrix_counts = confusion_matrix(y, predictions, labels=present)
    return {
        "report": report,
        "confusion_matrix": matrix_counts,
        "confusion_labels": [CLASSES[i] for i in present],
        "per_image": pd.DataFrame(fold_scores),
        "n_images": int(len(np.unique(group_values))),
        "n_candidates": int(len(y)),
    }

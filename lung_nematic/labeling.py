"""
Click-to-label widget for defect candidates.

Labels are made in context: the whole director field is shown with every
candidate marked, and each click cycles a candidate through
unlabelled -> real -> uncertain -> artefact -> unlabelled. Labelling against the
full field rather than isolated crops is deliberate - whether a +1/2 is real
often depends on its -1/2 partner two hundred microns away, which a crop hides.

The widget is matplotlib-only (works in Colab and Jupyter) and writes a plain
CSV: one row per candidate, keyed by the same ``candidate_id`` the feature
extractor produces, so labels join straight back onto features for training.

Typical use in a notebook cell::

    from lung_nematic.labeling import LabelingSession
    session = LabelingSession(rgb, field, candidates, image_id="23-15_1")
    session.show()
    # click to label, then:
    session.save("labels/23-15_1.csv")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# click cycle
LABEL_CYCLE = [None, "real", "uncertain", "artefact"]
LABEL_COLOUR = {
    None: "#3388FF",
    "real": "#00CC66",
    "uncertain": "#E6A000",
    "artefact": "#CC0000",
}
LABEL_MARKER = {0.5: "P", -0.5: "X", 1.0: "*", -1.0: "d"}


class LabelingSession:
    """Interactive labelling of one image's candidates."""

    def __init__(
        self,
        rgb: np.ndarray,
        field: dict[str, np.ndarray],
        candidates: pd.DataFrame,
        image_id: str,
        director_step_px: int = 45,
        pick_radius_px: float = 40.0,
    ):
        self.rgb = rgb
        self.field = field
        self.candidates = candidates.reset_index(drop=True)
        self.image_id = image_id
        self.director_step_px = director_step_px
        self.pick_radius_px = pick_radius_px

        self.labels: list = [None] * len(self.candidates)
        self._id_of = self._candidate_ids()

        self.figure = None
        self.axis = None
        self._markers = []

    def _candidate_ids(self) -> list[str]:
        ids = []
        for row in self.candidates.itertuples():
            ids.append(
                f"{int(round(row.x_px))}_{int(round(row.y_px))}_{row.charge:+.1f}"
            )
        return ids

    # ---------------------------------------------------------------- drawing
    def _draw_director(self) -> None:
        theta = self.field["theta"]
        order = self.field["order"]
        step = self.director_step_px
        height, width = theta.shape
        for y in range(step // 2, height, step):
            for x in range(step // 2, width, step):
                if order[y, x] < 0.1:
                    continue
                length = 6 + 20 * order[y, x]
                dx = length * np.cos(theta[y, x])
                dy = length * np.sin(theta[y, x])
                self.axis.plot([x - dx, x + dx], [y - dy, y + dy],
                               color="white", alpha=0.35, linewidth=0.6, zorder=2)

    def _draw_candidates(self) -> None:
        self._markers = []
        for index, row in self.candidates.iterrows():
            marker = LABEL_MARKER.get(float(row["charge"]), "o")
            handle = self.axis.scatter(
                row["x_px"], row["y_px"], s=140, marker=marker,
                facecolors="none", edgecolors=LABEL_COLOUR[self.labels[index]],
                linewidths=2.2, zorder=5, picker=False,
            )
            self._markers.append(handle)

    def _refresh_marker(self, index: int) -> None:
        self._markers[index].set_edgecolors(LABEL_COLOUR[self.labels[index]])
        self.figure.canvas.draw_idle()
        self._update_title()

    def _update_title(self) -> None:
        counts = {name: self.labels.count(name)
                  for name in ("real", "uncertain", "artefact")}
        remaining = self.labels.count(None)
        self.axis.set_title(
            f"{self.image_id}   |   "
            f"real {counts['real']}  uncertain {counts['uncertain']}  "
            f"artefact {counts['artefact']}  unlabelled {remaining}\n"
            f"click a marker to cycle: blue -> green(real) -> "
            f"amber(uncertain) -> red(artefact)",
            fontsize=10,
        )

    # ----------------------------------------------------------------- events
    def _on_click(self, event) -> None:
        if event.inaxes is not self.axis or event.xdata is None:
            return
        distances = np.hypot(
            self.candidates["x_px"].to_numpy() - event.xdata,
            self.candidates["y_px"].to_numpy() - event.ydata,
        )
        nearest = int(distances.argmin())
        if distances[nearest] > self.pick_radius_px:
            return
        current = LABEL_CYCLE.index(self.labels[nearest])
        self.labels[nearest] = LABEL_CYCLE[(current + 1) % len(LABEL_CYCLE)]
        self._refresh_marker(nearest)

    def show(self, figsize=(13, 10)) -> None:
        self.figure, self.axis = plt.subplots(figsize=figsize)
        self.axis.imshow(self.rgb, zorder=1)
        self._draw_director()
        self._draw_candidates()
        self._update_title()
        self.axis.set_xlim(0, self.rgb.shape[1])
        self.axis.set_ylim(self.rgb.shape[0], 0)
        self.axis.axis("off")
        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------- bulk shortcuts
    def label_all_unlabelled(self, label: str) -> None:
        """Set every still-unlabelled candidate at once (e.g. a frame of pure
        noise where all are artefacts)."""
        for index in range(len(self.labels)):
            if self.labels[index] is None:
                self.labels[index] = label
        if self.figure is not None:
            for index in range(len(self.labels)):
                self._refresh_marker(index)

    # ------------------------------------------------------------------ output
    def to_frame(self) -> pd.DataFrame:
        frame = self.candidates.copy()
        frame["candidate_id"] = self._id_of
        frame["image_id"] = self.image_id
        frame["label"] = self.labels
        return frame

    def save(self, path: str | Path, drop_unlabelled: bool = True) -> pd.DataFrame:
        frame = self.to_frame()
        if drop_unlabelled:
            frame = frame.loc[frame["label"].notna()]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        counts = frame["label"].value_counts().to_dict()
        print(f"saved {len(frame)} labels to {path}: {counts}")
        return frame


def load_labels(paths: list[str | Path]) -> pd.DataFrame:
    """Concatenate label CSVs from several images into one training table."""
    frames = [pd.read_csv(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.loc[combined["label"].notna()]
    return combined

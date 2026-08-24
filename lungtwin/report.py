"""Rendering of identifiability results as plain text.

The report is deliberately blunt about what cannot be estimated. The failure
mode this whole package exists to prevent is a plausible-looking fit to a model
whose parameters were never recoverable, so a report that buries the rank
deficiency in a table of numbers would defeat the purpose.
"""

from __future__ import annotations

import numpy as np

from .identifiability import IdentifiabilityReport

_INTERPRETATION = {
    frozenset({"beta"}): (
        "beta is not estimable from a patient whose treatment status never "
        "changes. Either restrict estimation to the pre-treatment window of "
        "patients who start therapy during follow-up, or fix beta with an "
        "informative prior from the antifibrotic trials and let the data "
        "update it weakly."
    ),
    frozenset({"r_i", "beta"}): (
        "only the difference r_i - beta is estimable under constant treatment "
        "exposure. Separating them requires a within-patient change of "
        "treatment status; a between-patient contrast will do it too, but that "
        "contrast carries confounding by indication."
    ),
    frozenset({"kappa", "dlco0"}): (
        "the DLCO channel is absent, so nothing that only enters through DLCO "
        "can be estimated. Drop kappa and dlco0 from the free set, or acquire "
        "DLCO."
    ),
    frozenset({"alpha"}): (
        "alpha has identically zero sensitivity: the reserve state does not "
        "reach any observation channel. Use ReserveMode.OBSERVED to give it "
        "one, or ReserveMode.FEEDBACK to let it act on the burden rate."
    ),
}


def render(report: IdentifiabilityReport, *, width: int = 78) -> str:
    lines: list[str] = []
    rule = "-" * width

    lines.append("IDENTIFIABILITY REPORT")
    lines.append(rule)
    lines.append(f"design       : {report.design}")
    lines.append(f"observations : {report.n_observations}")
    lines.append(f"parameters   : {len(report.free_names)} "
                 f"({', '.join(report.free_names)})")
    lines.append("")

    spectrum = np.array2string(
        report.singular_values, precision=4, suppress_small=True,
        max_line_width=width,
    )
    lines.append(f"singular values : {spectrum}")
    lines.append(f"numerical rank  : {report.rank} / {len(report.free_names)}")
    lines.append("")

    if report.is_structurally_identifiable:
        lines.append("STRUCTURAL: identifiable.")
        lines.append(f"  condition number {report.condition_number:.1f}")
        if report.condition_number > 1e4:
            lines.append(
                "  WARNING: severe ill-conditioning. The model is identifiable "
                "in exact arithmetic only; treat it as unidentifiable in "
                "practice."
            )
    else:
        deficiency = len(report.free_names) - report.rank
        lines.append(
            f"STRUCTURAL: NOT identifiable - {deficiency} null direction(s)."
        )
        lines.append(
            "  No amount of data, no prior and no optimiser recovers these "
            "combinations. Fitting anyway returns whatever point the "
            "initialisation drifted to."
        )
        for index, direction in enumerate(report.null_directions, start=1):
            terms = "  ".join(
                f"{name}={value:+.3f}" for name, value in direction.items()
            )
            lines.append(f"  [{index}] {terms}")
            note = _INTERPRETATION.get(frozenset(direction))
            if note:
                for wrapped in _wrap(note, width - 6):
                    lines.append(f"      {wrapped}")

    lines.append("")
    if report.standard_errors is not None:
        lines.append("PRACTICAL: Cramer-Rao lower bounds (pinned noise)")
        for name, error in report.standard_errors.items():
            lines.append(f"  SE({name}) >= {error:.3f}")
        lines.append("")
        lines.append(
            "  These are lower bounds and they are local. Compare against the "
            "effect size you need to detect: an SE on r_i larger than r_i "
            "itself means the design cannot distinguish a progressing patient "
            "from a stable one."
        )
        if report.correlations is not None:
            worst = _worst_correlation(report)
            if worst is not None:
                first, second, value = worst
                lines.append(
                    f"  strongest correlation: {first} vs {second} = "
                    f"{value:+.3f}"
                )
                if abs(value) > 0.95:
                    lines.append(
                        "  WARNING: near-collinear. The pair is jointly "
                        "constrained but individually poorly determined."
                    )
    else:
        lines.append(
            "PRACTICAL: not computed - the information matrix is singular. "
            "Fix the structural problem first; a precision bound on an "
            "unidentifiable model is meaningless."
        )

    lines.append(rule)
    return "\n".join(lines)


def _worst_correlation(report: IdentifiabilityReport):
    matrix = report.correlations
    if matrix is None or matrix.shape[0] < 2:
        return None
    offdiag = np.abs(matrix - np.eye(matrix.shape[0]))
    flat = int(np.argmax(offdiag))
    row, column = np.unravel_index(flat, offdiag.shape)
    return (
        report.free_names[row],
        report.free_names[column],
        float(matrix[row, column]),
    )


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width) or [""]

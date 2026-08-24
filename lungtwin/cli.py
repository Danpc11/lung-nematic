"""Command line entry point: analyse a candidate study design before fitting."""

from __future__ import annotations

import argparse

from .design import routine_followup
from .identifiability import analyze, monte_carlo_check, visits_required
from .model import Channel, Parameters, ReserveMode
from .report import render

CORE = ["fvc0", "dlco0", "r_i", "kappa", "beta"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lungtwin-ident",
        description=(
            "Structural and practical identifiability for the two-state IPF "
            "twin. Run this before writing an estimator: a fit to a "
            "structurally unidentifiable model does not fail, it just returns "
            "an arbitrary point."
        ),
    )
    parser.add_argument("--visits", type=int, default=4,
                        help="Number of visits (default 4).")
    parser.add_argument("--interval-months", type=float, default=6.0,
                        help="Months between visits (default 6).")
    parser.add_argument(
        "--treatment-start-months", type=float, default=None,
        help=(
            "When an antifibrotic is started, in months from baseline. Omit "
            "for never treated, 0 for treated from baseline. This single "
            "choice decides whether beta is estimable at all."
        ),
    )
    parser.add_argument("--no-dlco", action="store_true",
                        help="Drop the DLCO channel entirely.")
    parser.add_argument(
        "--dlco-missing-rate", type=float, default=0.0,
        help=(
            "Fraction of visits with DLCO missing completely at random. Real "
            "DLCO missingness is informative; this models the precision loss "
            "only, never the bias."
        ),
    )
    parser.add_argument(
        "--fix-beta", action="store_true",
        help=(
            "Treat beta as known (pinned by an informative prior from the "
            "antifibrotic trials) rather than estimating it."
        ),
    )
    parser.add_argument(
        "--reserve", choices=[m.value for m in ReserveMode],
        default=ReserveMode.NONE.value,
        help=(
            "How the reserve state is wired. 'none' drops it (and makes the "
            "model reduce to per-patient linear regression); 'observed' gives "
            "it an SpO2 channel; 'feedback' lets it modulate the burden rate."
        ),
    )
    parser.add_argument("--monte-carlo", type=int, default=0,
                        help="Validate the CRLB against N refitted replicates.")
    parser.add_argument(
        "--target-se-ri", type=float, default=None,
        help="Report the visit count needed for this SE on r_i.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    mode = ReserveMode(args.reserve)
    channels: tuple[Channel, ...] = (
        (Channel.FVC,) if args.no_dlco else (Channel.FVC, Channel.DLCO)
    )
    if mode is ReserveMode.OBSERVED:
        channels = channels + (Channel.SPO2,)

    free = [name for name in CORE if not (args.fix_beta and name == "beta")]
    if args.no_dlco:
        free = [name for name in free if name not in {"kappa", "dlco0"}]
    if mode is ReserveMode.OBSERVED:
        free = free + ["alpha", "spo2_0", "lam"]
    elif mode is ReserveMode.FEEDBACK:
        free = free + ["alpha", "gamma"]

    treatment_start = (
        None if args.treatment_start_months is None
        else args.treatment_start_months / 12.0
    )

    params = Parameters(gamma=0.8 if mode is ReserveMode.FEEDBACK else 0.0)
    schedule = routine_followup(
        n_visits=args.visits,
        interval_months=args.interval_months,
        treatment_start=treatment_start,
        channels=channels,
        dlco_missing_rate=args.dlco_missing_rate,
        seed=args.seed,
    )

    report = analyze(params, free, schedule, mode)
    print(render(report))

    if args.target_se_ri is not None:
        found = visits_required(
            params, free, {"r_i": args.target_se_ri}, mode=mode,
            interval_months=args.interval_months,
            treatment_start=treatment_start, channels=channels,
        )
        needed = found["r_i"]
        print()
        if needed is None:
            print(
                f"DESIGN: SE(r_i) <= {args.target_se_ri} is unreachable at any "
                "follow-up length with this design."
            )
            if not args.fix_beta:
                print(
                    "  beta is informed only by the pre-treatment window, so "
                    "visits after the switch barely separate it from r_i. Try "
                    "--fix-beta, or a design with a longer untreated window."
                )
        else:
            years = (needed - 1) * args.interval_months / 12.0
            print(
                f"DESIGN: {needed} visits ({years:.1f} years) reach "
                f"SE(r_i) <= {args.target_se_ri}."
            )

    if args.monte_carlo > 0 and report.is_structurally_identifiable:
        print()
        result = monte_carlo_check(params, free, schedule, mode,
                                   n_replicates=args.monte_carlo,
                                   seed=args.seed)
        print(f"MONTE CARLO ({result['n_replicates']} replicates)")
        print(f"{'param':>8}  {'CRLB':>9}  {'robust SD':>9}  {'plain SD':>11}")
        for name in free:
            print(
                f"{name:>8}  {result['crlb'][name]:9.3f}  "
                f"{result['robust_sd'][name]:9.3f}  "
                f"{result['empirical_sd'][name]:11.3f}"
            )
        if result["collapse_fraction"] > 0:
            print(
                f"  {result['collapse_fraction']:.1%} of replicates fitted a "
                "near-flat trajectory. In those, any parameter that only "
                "multiplies the burden (kappa) is unconstrained - which is why "
                "plain SD can exceed the CRLB by orders of magnitude while the "
                "robust SD still agrees with it."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line interface for the fibroblastic-focus simulation."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

from .bistability import critical_value, fixed_points, scan_two_parameters
from .model import FocusConfig
from .render import run_and_record

# Parameters exposed on the command line: everything that changes the science.
TUNABLE = [
    "total_time_h", "injury_duration_h",
    "deposition_rate_kPa_per_h", "degradation_rate_per_h",
    "memory_factor", "activation_rate_per_h", "deactivation_rate_per_h",
    "E_act_kPa", "E_tgfb_kPa", "E_healthy_kPa", "E_max_kPa",
    "injury_provisional_E_kPa", "injury_radius_um", "injury_activation_drop_kPa",
    "durotaxis_um2_per_kPa_h", "speed_um_per_h", "speed_myo_factor",
    "align_rate_per_h", "rot_diffusion_per_h", "prolif_rate_per_h",
    "n_initial", "width_um", "height_um", "seed",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Agent-based active-nematic simulation of fibroblastic focus "
            "formation, with bistability analysis of the point of no return."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_tunables(target: argparse.ArgumentParser) -> None:
        types = {f.name: f.type for f in fields(FocusConfig)}
        for name in TUNABLE:
            kind = int if types[name] is int else float
            target.add_argument(f"--{name.replace('_', '-')}", type=kind,
                                default=None, dest=name)

    run = sub.add_parser("run", help="Run a scenario and write GIF/MP4.")
    run.add_argument("--output", required=True)
    run.add_argument("--frame-every-h", type=float, default=8.0)
    run.add_argument("--fps", type=int, default=10)
    run.add_argument("--no-defects", action="store_true")
    run.add_argument("--no-gif", action="store_true")
    run.add_argument("--no-mp4", action="store_true")
    add_tunables(run)

    crit = sub.add_parser("critical",
                          help="Bisect the no-return value of one parameter.")
    crit.add_argument("--parameter", required=True)
    crit.add_argument("--low", type=float, required=True)
    crit.add_argument("--high", type=float, required=True)
    add_tunables(crit)

    scan = sub.add_parser("scan", help="Two-parameter phase diagram (CSV).")
    scan.add_argument("--x", required=True)
    scan.add_argument("--x-range", nargs=3, type=float, required=True,
                      metavar=("LOW", "HIGH", "N"))
    scan.add_argument("--y", required=True)
    scan.add_argument("--y-range", nargs=3, type=float, required=True,
                      metavar=("LOW", "HIGH", "N"))
    scan.add_argument("--output", required=True)
    add_tunables(scan)

    return parser


def _config_from_args(args) -> FocusConfig:
    overrides = {
        name: getattr(args, name)
        for name in TUNABLE
        if getattr(args, name, None) is not None
    }
    config = replace(FocusConfig(), **overrides)
    config.validate()
    return config


def main() -> None:
    args = build_parser().parse_args()
    config = _config_from_args(args)

    if args.command == "run":
        outputs = run_and_record(
            config, args.output,
            frame_every_h=args.frame_every_h, fps=args.fps,
            make_gif=not args.no_gif, make_mp4=not args.no_mp4,
            show_defects=not args.no_defects,
        )
        print(json.dumps({k: v for k, v in outputs.items() if k != "final"},
                         indent=2))
        final = outputs["final"]
        print(f"\nFinal state at day {final['time_d']:.1f}:")
        print(f"  lesion stiffness   {final['E_focus_kPa']:.1f} kPa")
        print(f"  myofibroblasts     {final['myo_fraction'] * 100:.0f}%")
        print(f"  defects            +{final['n_plus']} / -{final['n_minus']}")
        verdict = ("PERSISTENT focus (past the point of no return)"
                   if final["E_focus_kPa"] > config.E_act_kPa
                   else "lesion resolving")
        print(f"  verdict            {verdict}")

    elif args.command == "critical":
        value = critical_value(config, args.parameter, args.low, args.high)
        info = fixed_points(config)
        print(json.dumps({
            "parameter": args.parameter,
            "critical_value": value,
            "bistable_at_default": info["bistable"],
            "E_separatrix_kPa": info["E_separatrix"],
            "E_fibrotic_kPa": info["E_fibrotic"],
        }, indent=2))

    elif args.command == "scan":
        x_low, x_high, x_n = args.x_range
        y_low, y_high, y_n = args.y_range
        frame = scan_two_parameters(
            config,
            args.x, np.linspace(x_low, x_high, int(x_n)),
            args.y, np.linspace(y_low, y_high, int(y_n)),
        )
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        persisted = frame["persisted"].mean()
        print(f"Wrote {path} ({len(frame)} points, "
              f"{persisted * 100:.0f}% persistent).")


if __name__ == "__main__":
    main()

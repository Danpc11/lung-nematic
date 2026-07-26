"""Command-line entry point for the small true-3D alveolar simulation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json

from .model import accelerated_3d_demo_config, human_chronic_3d_config
from .render import Alveolar3DRenderConfig, run_and_record_3d


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate a small cluster of alveoli with true xyz cell dynamics, "
            "respiration, collapse and fibroblastic-focus formation."
        )
    )
    parser.add_argument("--output", default="alveolar_3d_output")
    parser.add_argument(
        "--preset",
        choices=("accelerated", "human-chronic"),
        default="accelerated",
        help=(
            "accelerated is the short visual demo; human-chronic uses a "
            "clinical-scale clock calibrated over two years."
        ),
    )
    parser.add_argument(
        "--days",
        type=float,
        default=None,
        help="Calendar duration (default: 12 or 730.5, depending on preset).",
    )
    parser.add_argument(
        "--state-every-hours",
        type=float,
        default=None,
        help="Calendar interval between saved states (default: 1 or 30.4 days).",
    )
    parser.add_argument("--breathing-frames", type=int, default=6)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument(
        "--rate-scale",
        type=float,
        default=None,
        help="Override the preset's biological-rate multiplier.",
    )
    parser.add_argument("--orbit-degrees", type=float, default=2.1)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--no-mp4", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.days is not None and args.days <= 0:
        parser.error("--days must be positive")
    if args.breathing_frames < 2:
        parser.error("--breathing-frames must be at least 2")
    if args.rate_scale is not None and args.rate_scale <= 0:
        parser.error("--rate-scale must be positive")

    if args.preset == "human-chronic":
        config = human_chronic_3d_config(seed=args.seed)
        default_days = config.total_time_h / 24.0
        default_state_every_h = 30.4375 * 24.0
    else:
        config = accelerated_3d_demo_config(seed=args.seed)
        default_days = config.total_time_h / 24.0
        default_state_every_h = 24.0

    config = replace(
        config,
        total_time_h=24.0 * (
            args.days if args.days is not None else default_days
        ),
        kinetic_rate_scale=(
            args.rate_scale
            if args.rate_scale is not None
            else config.kinetic_rate_scale
        ),
    )
    render = Alveolar3DRenderConfig(
        breathing_frames_per_cycle=args.breathing_frames,
        orbit_deg_per_frame=args.orbit_degrees / args.breathing_frames,
    )
    outputs = run_and_record_3d(
        config,
        args.output,
        state_every_h=(
            args.state_every_hours
            if args.state_every_hours is not None
            else default_state_every_h
        ),
        breathing_subframes=args.breathing_frames,
        fps=args.fps,
        render_config=render,
        make_gif=not args.no_gif,
        make_mp4=not args.no_mp4,
        progress=lambda count: print(f"rendered frame {count}", flush=True),
    )
    printable = {
        key: value
        for key, value in outputs.items()
        if key != "simulation"
    }
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

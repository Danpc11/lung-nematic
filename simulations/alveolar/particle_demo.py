"""Command-line demo for the 2.5D alveolar particle simulation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json

from .particle_render import (
    ParticleRenderConfig,
    accelerated_particle_demo_config,
    run_and_record_particles,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an accelerated multicellular alveolar simulation and render "
            "epithelial cells, fibroblasts and myofibroblasts as particles."
        )
    )
    parser.add_argument("--output", default="particle_demo_output")
    parser.add_argument("--days", type=float, default=15.0)
    parser.add_argument("--frame-every-hours", type=float, default=18.0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--breathing-frames",
        type=int,
        default=6,
        help="Visual subframes per biological state, spanning one breath.",
    )
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--orbit-degrees", type=float, default=1.2)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--no-mp4", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.breathing_frames < 2:
        parser.error("--breathing-frames must be at least 2")
    config = replace(
        accelerated_particle_demo_config(seed=args.seed),
        total_time_h=24.0 * args.days,
    )
    render = ParticleRenderConfig(
        orbit_deg_per_frame=args.orbit_degrees / args.breathing_frames,
        breathing_frames_per_cycle=args.breathing_frames,
    )
    outputs = run_and_record_particles(
        config,
        args.output,
        frame_every_h=args.frame_every_hours,
        fps=args.fps,
        breathing_subframes=args.breathing_frames,
        render_config=render,
        make_gif=not args.no_gif,
        make_mp4=not args.no_mp4,
        progress=lambda count: print(f"rendered frame {count}", flush=True),
    )
    printable = {key: value for key, value in outputs.items() if key != "simulation"}
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

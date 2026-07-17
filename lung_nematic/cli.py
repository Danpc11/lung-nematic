from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .batch import analyze_folder, summarize_by_group
from .config import load_config, load_default_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze nematic organization and candidate topological "
            "defects in lung histology images."
        )
    )

    parser.add_argument("--input", required=True,
                        help="Directory containing histology images.")
    parser.add_argument("--output", required=True,
                        help="Directory where results will be written.")
    parser.add_argument(
        "--config", default=None,
        help="JSON configuration file. Omit to use the packaged default.",
    )
    parser.add_argument("--metadata", default=None,
                        help="Optional CSV file containing image metadata.")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop when an image cannot be processed.")

    # Analysis selection (override the config when provided).
    parser.add_argument("--field", choices=("nuclear", "collagen", "fused"),
                        default=None, help="Orientation source for the field.")
    parser.add_argument("--run-null", action=argparse.BooleanOptionalAction,
                        default=None, help="Run the permutation null model.")
    parser.add_argument("--run-colocalization",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Run the colocalization test.")
    parser.add_argument("--n-permutations", type=int, default=None)
    parser.add_argument("--null-mode", choices=("shuffle", "uniform"),
                        default=None)
    parser.add_argument("--null-downsample", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--collagen-inner-scale", type=float, default=None)
    parser.add_argument("--mask-normalized-smoothing",
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for all stochastic controls.")

    return parser


def _apply_overrides(config, args):
    mapping = {
        "field_type": args.field,
        "run_null": args.run_null,
        "run_colocalization": args.run_colocalization,
        "n_permutations": args.n_permutations,
        "null_mode": args.null_mode,
        "null_downsample": args.null_downsample,
        "n_bootstrap": args.n_bootstrap,
        "collagen_inner_scale_px": args.collagen_inner_scale,
        "mask_normalized_smoothing": args.mask_normalized_smoothing,
        "random_seed": args.seed,
    }
    overrides = {k: v for k, v in mapping.items() if v is not None}
    if overrides:
        config = replace(config, **overrides)
    config.validate()
    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        parser.error(f"Input directory does not exist: {input_dir}")

    if args.config is None:
        config = load_default_config()
    else:
        config_path = Path(args.config)
        if not config_path.exists():
            parser.error(f"Configuration file does not exist: {config_path}")
        config = load_config(config_path)

    config = _apply_overrides(config, args)

    summary, errors = analyze_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        metadata_csv=args.metadata,
        config=config,
        continue_on_error=not args.stop_on_error,
    )

    print(f"\nSuccessfully processed images: {len(summary)}")
    print(f"Failed images: {len(errors)}")
    print(f"Results directory: {output_dir.resolve()}")

    if not summary.empty:
        summarize_by_group(summary).to_csv(output_dir / "group_summary.csv")

    if not errors.empty:
        print(
            "\nSome images failed. Review:"
            f"\n{output_dir / 'processing_errors.csv'}"
        )


if __name__ == "__main__":
    main()

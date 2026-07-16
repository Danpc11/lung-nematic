from __future__ import annotations

import argparse
from pathlib import Path

from .batch import analyze_folder, summarize_by_group
from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze nematic organization and candidate topological "
            "defects in lung histology images."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing histology images.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where results will be written.",
    )
    parser.add_argument(
        "--config",
        default="config/default_config.json",
        help="Path to the JSON configuration file.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional CSV file containing image metadata.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop when an image cannot be processed.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    config_path = Path(args.config)

    if not input_dir.exists():
        parser.error(f"Input directory does not exist: {input_dir}")

    if not config_path.exists():
        parser.error(f"Configuration file does not exist: {config_path}")

    config = load_config(config_path)

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
        group_summary = summarize_by_group(summary)
        group_summary.to_csv(
            output_dir / "group_summary.csv"
        )

    if not errors.empty:
        print(
            "\nSome images failed. Review:"
            f"\n{output_dir / 'processing_errors.csv'}"
        )


if __name__ == "__main__":
    main()

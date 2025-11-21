
from __future__ import annotations

import argparse
from pathlib import Path

from ai_tutor.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline QLoRA fine-tuning script.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Config.lora_adapter_path),
        help="Directory to save LoRA adapter weights.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum number of training steps (keep small for experimentation).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do not actually train; just print what would happen.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== QLoRA Fine-Tuning Script ===")
    print(f"Base model ID:      {Config.base_model_id}")
    print(f"Base model path:    {Config.base_model_path}")
    print(f"LoRA adapter path:  {output_dir}")
    print(f"Max steps:          {args.max_steps}")
    print(f"Dry run:            {args.dry_run}")
    print()

    if args.dry_run:
        print("Dry run enabled. No training will be performed.")
        return

    # TODO: Implement QLoRA training pipeline here.
    print("QLoRA training is not yet implemented.")
    print("Implement training logic when you are ready to run offline fine-tuning.")


if __name__ == "__main__":
    main()

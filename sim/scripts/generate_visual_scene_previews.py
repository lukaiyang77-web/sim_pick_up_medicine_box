from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from visual_scene_utils import DEFAULT_CONDITION_TARGETS, DEFAULT_OUTPUT_DIR, expected_preview_paths, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all visual scene v1 preview images.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_textures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution_width", type=int, default=1024)
    parser.add_argument("--resolution_height", type=int, default=768)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve().parent / "create_textured_medicine_scene.py"
    preview_paths = expected_preview_paths(output_dir)

    for index, (condition, target_class) in enumerate(DEFAULT_CONDITION_TARGETS.items()):
        image_path, metadata_path = preview_paths[condition]
        command = [
            sys.executable,
            str(script_path),
            "--condition",
            condition,
            "--target_class",
            target_class,
            "--output_image",
            str(image_path),
            "--output_metadata",
            str(metadata_path),
            "--seed",
            str(args.seed + index),
            "--resolution_width",
            str(args.resolution_width),
            "--resolution_height",
            str(args.resolution_height),
        ]
        command.append("--headless" if args.headless else "--no-headless")
        command.append("--use_textures" if args.use_textures else "--no-use_textures")

        print(f"Generating {condition} -> {image_path}")
        subprocess.run(command, check=True)

    print(f"Generated previews in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

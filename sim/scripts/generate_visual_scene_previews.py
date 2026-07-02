from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from visual_scene_utils import CAMERA_MODES, DEFAULT_CONDITION_TARGETS, DEFAULT_OUTPUT_DIR, expected_preview_paths, resolve_project_path


DEBUG_TOPDOWN_TARGETS = {
    "front": DEFAULT_CONDITION_TARGETS["front"],
    "similar_distractor": DEFAULT_CONDITION_TARGETS["similar_distractor"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all visual scene v1 preview images.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_textures", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--camera_mode", choices=sorted(CAMERA_MODES), default="oblique")
    parser.add_argument("--box_scale", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution_width", type=int, default=1024)
    parser.add_argument("--resolution_height", type=int, default=768)
    return parser.parse_args()


def run_scene(
    script_path: Path,
    condition: str,
    target_class: str,
    image_path: Path,
    metadata_path: Path,
    seed: int,
    args: argparse.Namespace,
    camera_mode: str,
) -> None:
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
        str(seed),
        "--camera_mode",
        camera_mode,
        "--box_scale",
        str(args.box_scale),
        "--resolution_width",
        str(args.resolution_width),
        "--resolution_height",
        str(args.resolution_height),
    ]
    command.append("--headless" if args.headless else "--no-headless")
    command.append("--use_textures" if args.use_textures else "--no-use_textures")

    print(f"Generating {condition} ({camera_mode}) -> {image_path}")
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve().parent / "create_textured_medicine_scene.py"
    preview_paths = expected_preview_paths(output_dir)

    for index, (condition, target_class) in enumerate(DEFAULT_CONDITION_TARGETS.items()):
        image_path, metadata_path = preview_paths[condition]
        run_scene(
            script_path,
            condition,
            target_class,
            image_path,
            metadata_path,
            args.seed + index,
            args,
            args.camera_mode,
        )

    for index, (condition, target_class) in enumerate(DEBUG_TOPDOWN_TARGETS.items()):
        run_scene(
            script_path,
            condition,
            target_class,
            output_dir / f"debug_topdown_{condition}.png",
            output_dir / f"debug_topdown_{condition}.json",
            args.seed + 100 + index,
            args,
            "top_down",
        )

    print(f"Generated previews in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

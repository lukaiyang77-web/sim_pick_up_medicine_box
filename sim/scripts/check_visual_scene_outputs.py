from __future__ import annotations

import argparse
import json
from pathlib import Path

from visual_scene_utils import CONDITIONS, DEFAULT_OUTPUT_DIR, expected_preview_paths, resolve_project_path


REQUIRED_METADATA_KEYS = {
    "condition",
    "target_class",
    "objects",
    "camera_pose",
    "camera_mode",
    "bin_pose",
    "use_textures",
}

REQUIRED_OBJECT_KEYS = {
    "class_name",
    "instance_id",
    "pose",
    "dimensions",
    "is_target",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check visual scene v1 preview outputs.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def load_json(path: Path, errors: list[str]) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{path.name}: failed to read JSON: {exc}")
        return None


def validate_metadata(condition: str, metadata: dict, errors: list[str]) -> None:
    missing = REQUIRED_METADATA_KEYS - set(metadata)
    if missing:
        errors.append(f"{condition}: metadata missing keys: {sorted(missing)}")
        return

    if metadata["condition"] != condition:
        errors.append(f"{condition}: metadata condition is {metadata['condition']!r}")
    if metadata.get("camera_mode") not in {"oblique", "top_down"}:
        errors.append(f"{condition}: invalid camera_mode {metadata.get('camera_mode')!r}")

    objects = metadata.get("objects")
    if not isinstance(objects, list):
        errors.append(f"{condition}: objects is not a list")
        return

    if not 3 <= len(objects) <= 5:
        errors.append(f"{condition}: expected 3 to 5 medicine boxes, found {len(objects)}")

    class_names = [obj.get("class_name") for obj in objects if isinstance(obj, dict)]
    if len(class_names) != len(set(class_names)):
        errors.append(f"{condition}: duplicate class_name values found: {class_names}")

    target_class = metadata.get("target_class")
    target_occurrences = sum(1 for obj in objects if obj.get("class_name") == target_class)
    is_target_occurrences = sum(1 for obj in objects if obj.get("is_target") is True)
    if target_occurrences != 1:
        errors.append(f"{condition}: target_class {target_class!r} appears {target_occurrences} times")
    if is_target_occurrences != 1:
        errors.append(f"{condition}: expected exactly one is_target object, found {is_target_occurrences}")

    if condition == "similar_distractor":
        required = {"amoxicillin", "cefixime"}
        if not required.issubset(set(class_names)):
            errors.append("similar_distractor: missing amoxicillin and cefixime pair")

    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            errors.append(f"{condition}: object {index} is not a JSON object")
            continue
        missing_obj = REQUIRED_OBJECT_KEYS - set(obj)
        if missing_obj:
            errors.append(f"{condition}: object {index} missing keys: {sorted(missing_obj)}")


def check_outputs(output_dir: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    lines: list[str] = [f"Checking visual scene v1 outputs in {output_dir}"]

    for condition in CONDITIONS:
        image_path, metadata_path = expected_preview_paths(output_dir)[condition]
        if not image_path.exists():
            errors.append(f"{condition}: missing PNG {image_path.name}")
        elif image_path.stat().st_size <= 0:
            errors.append(f"{condition}: PNG is empty {image_path.name}")
        else:
            lines.append(f"OK PNG: {image_path.name} ({image_path.stat().st_size} bytes)")

        if not metadata_path.exists():
            errors.append(f"{condition}: missing JSON {metadata_path.name}")
            continue

        metadata = load_json(metadata_path, errors)
        if metadata is not None:
            validate_metadata(condition, metadata, errors)
            lines.append(f"OK JSON: {metadata_path.name}")

    if errors:
        lines.append("")
        lines.append("FAILED")
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("")
        lines.append("PASSED: all 6 visual scene previews and metadata files are complete.")

    return lines, errors


def main() -> int:
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines, errors = check_outputs(output_dir)
    summary_path = output_dir / "check_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Summary written to {summary_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    "capture_mode",
    "camera_prim_path",
    "bin_pose",
    "use_textures",
}

REQUIRED_OBJECT_KEYS = {
    "class_name",
    "instance_id",
    "pose",
    "dimensions",
    "is_target",
    "prim_path",
    "stage_prim_exists",
    "actual_translation",
    "actual_scale",
    "used_texture",
    "texture_binding_method",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check visual scene v1 preview outputs.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--require_textures", action="store_true")
    return parser.parse_args()


def load_json(path: Path, errors: list[str]) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{path.name}: failed to read JSON: {exc}")
        return None


def validate_metadata(condition: str, metadata: dict, errors: list[str], texture_stats: dict) -> None:
    missing = REQUIRED_METADATA_KEYS - set(metadata)
    if missing:
        errors.append(f"{condition}: metadata missing keys: {sorted(missing)}")
        return

    if metadata["condition"] != condition:
        errors.append(f"{condition}: metadata condition is {metadata['condition']!r}")
    if metadata.get("camera_mode") not in {"oblique", "top_down"}:
        errors.append(f"{condition}: invalid camera_mode {metadata.get('camera_mode')!r}")
    if metadata.get("capture_mode") not in {"camera_sensor", "replicator", "viewport"}:
        errors.append(f"{condition}: invalid capture_mode {metadata.get('capture_mode')!r}")
    if not metadata.get("camera_prim_path"):
        errors.append(f"{condition}: missing camera_prim_path")

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
        if obj.get("stage_prim_exists") is not True:
            errors.append(f"{condition}: object {index} stage_prim_exists is not true")
        if obj.get("prim_type_name") != "Cube":
            errors.append(f"{condition}: object {index} prim_type_name is not Cube")
        if not obj.get("actual_translation"):
            errors.append(f"{condition}: object {index} missing actual_translation")
        if not obj.get("actual_scale"):
            errors.append(f"{condition}: object {index} missing actual_scale")
        if obj.get("used_texture") is True:
            texture_stats["texture_success_count"] += 1
        else:
            texture_stats["texture_fallback_count"] += 1
            class_name = obj.get("class_name", "unknown")
            texture_stats["fallback_classes"].add(class_name)
            texture_stats["warnings"].append(
                f"{condition}: {class_name} used fallback material "
                f"({obj.get('texture_error') or obj.get('texture_binding_method')})"
            )


def validate_debug_big_cube(output_dir: Path, errors: list[str], lines: list[str]) -> None:
    image_path = output_dir / "debug_big_cube.png"
    metadata_path = output_dir / "debug_big_cube.json"
    if not image_path.exists() and not metadata_path.exists():
        return
    if not image_path.exists():
        errors.append("debug_big_cube: missing PNG debug_big_cube.png")
    elif image_path.stat().st_size <= 0:
        errors.append("debug_big_cube: PNG is empty debug_big_cube.png")
    else:
        lines.append(f"OK PNG: {image_path.name} ({image_path.stat().st_size} bytes)")

    if not metadata_path.exists():
        errors.append("debug_big_cube: missing JSON debug_big_cube.json")
        return

    metadata = load_json(metadata_path, errors)
    if metadata is None:
        return
    debug_cube = metadata.get("debug_big_cube")
    if not isinstance(debug_cube, dict):
        errors.append("debug_big_cube: metadata missing debug_big_cube object")
        return
    if debug_cube.get("stage_prim_exists") is not True:
        errors.append("debug_big_cube: DebugBigRedCube stage_prim_exists is not true")
    if debug_cube.get("prim_path") != "/World/DebugBigRedCube":
        errors.append(f"debug_big_cube: unexpected prim_path {debug_cube.get('prim_path')!r}")
    lines.append("OK JSON: debug_big_cube.json")


def check_outputs(output_dir: Path, require_textures: bool = False) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    lines: list[str] = [f"Checking visual scene v1 outputs in {output_dir}"]
    texture_stats = {
        "texture_success_count": 0,
        "texture_fallback_count": 0,
        "fallback_classes": set(),
        "warnings": [],
    }

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
            validate_metadata(condition, metadata, errors, texture_stats)
            lines.append(f"OK JSON: {metadata_path.name}")

    validate_debug_big_cube(output_dir, errors, lines)

    lines.append("")
    lines.append("Texture summary")
    lines.append(f"- require_textures: {require_textures}")
    lines.append(f"- texture_success_count: {texture_stats['texture_success_count']}")
    lines.append(f"- texture_fallback_count: {texture_stats['texture_fallback_count']}")
    fallback_classes = sorted(texture_stats["fallback_classes"])
    lines.append(f"- fallback_classes: {fallback_classes}")
    if require_textures and texture_stats["warnings"]:
        lines.append("- texture warnings:")
        lines.extend(f"  - {warning}" for warning in texture_stats["warnings"])

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
    lines, errors = check_outputs(output_dir, args.require_textures)
    summary_path = output_dir / "check_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Summary written to {summary_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MEDICINE_CLASSES = [
    "amoxicillin",
    "cefixime",
    "ibuprofen",
    "montmorillonite",
    "vitamin_c",
]

CONDITIONS = [
    "front",
    "back",
    "side",
    "occlusion_25",
    "occlusion_50",
    "similar_distractor",
]

DEFAULT_CONDITION_TARGETS = {
    "front": "amoxicillin",
    "back": "cefixime",
    "side": "ibuprofen",
    "occlusion_25": "montmorillonite",
    "occlusion_50": "vitamin_c",
    "similar_distractor": "amoxicillin",
}

BOX_DIMENSIONS = {
    "length": 0.12,
    "width": 0.07,
    "height": 0.035,
}

CLASS_COLORS = {
    "amoxicillin": (0.90, 0.18, 0.16),
    "cefixime": (0.12, 0.40, 0.92),
    "ibuprofen": (0.16, 0.68, 0.32),
    "montmorillonite": (0.92, 0.70, 0.20),
    "vitamin_c": (0.92, 0.36, 0.08),
}

TABLE_POSE = {
    "position": [0.0, 0.0, 0.73],
    "dimensions": [0.80, 0.60, 0.04],
}

BIN_POSE = {
    "position": [0.25, 0.12, 0.755],
    "dimensions": [0.24, 0.18, 0.01],
    "color": [0.35, 0.78, 0.86],
}

CAMERA_POSE = {
    "position": [0.0, -0.78, 1.12],
    "look_at": [0.0, 0.0, 0.76],
    "fov_degrees": 45.0,
}

DEFAULT_OUTPUT_DIR = Path("sim/results/visual_scene_v1")


@dataclass(frozen=True)
class SceneObjectSpec:
    class_name: str
    instance_id: str
    pose: dict
    dimensions: dict
    condition: str
    is_target: bool
    material_color: tuple[float, float, float]
    texture_path: str | None = None
    used_texture: bool = False

    def to_metadata(self) -> dict:
        return {
            "class_name": self.class_name,
            "instance_id": self.instance_id,
            "pose": self.pose,
            "dimensions": self.dimensions,
            "condition": self.condition,
            "is_target": self.is_target,
            "material_color": list(self.material_color),
            "texture_path": self.texture_path,
            "used_texture": self.used_texture,
        }


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def ensure_supported(condition: str, target_class: str) -> None:
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition}. Expected one of {CONDITIONS}.")
    if target_class not in MEDICINE_CLASSES:
        raise ValueError(
            f"Unsupported target_class: {target_class}. Expected one of {MEDICINE_CLASSES}."
        )


def dimensions_dict() -> dict:
    return dict(BOX_DIMENSIONS)


def object_z() -> float:
    return 0.75 + BOX_DIMENSIONS["height"] / 2.0


def yaw_for_condition(condition: str) -> float:
    if condition == "back":
        return 180.0
    if condition == "side":
        return 90.0
    return 0.0


def front_texture_candidates(class_name: str) -> list[Path]:
    base = project_root() / "data" / "textures" / "medicine_box_images_v3" / class_name
    return [
        base / "views" / "front.png",
        base / "raw" / "front_box.png",
        base / "raw" / "front_blue.png",
        base / "raw" / "front_red.png",
        base / "front.png",
    ]


def find_front_texture(class_name: str) -> Path | None:
    for candidate in front_texture_candidates(class_name):
        if candidate.exists() and candidate.is_file():
            return candidate
    base = project_root() / "data" / "textures" / "medicine_box_images_v3" / class_name
    if not base.exists():
        return None
    matches = sorted(base.glob("**/front*.png"))
    return matches[0] if matches else None


def select_unique_classes(
    condition: str,
    target_class: str,
    rng: random.Random,
) -> list[str]:
    ensure_supported(condition, target_class)

    if condition == "similar_distractor":
        selected = [target_class]
        for class_name in ("amoxicillin", "cefixime"):
            if class_name not in selected:
                selected.append(class_name)
        for class_name in MEDICINE_CLASSES:
            if len(selected) >= 3:
                break
            if class_name not in selected:
                selected.append(class_name)
        return selected

    distractors = [name for name in MEDICINE_CLASSES if name != target_class]
    rng.shuffle(distractors)
    return [target_class, *distractors[:2]]


def condition_positions(condition: str, classes: Iterable[str], target_class: str) -> dict[str, tuple[float, float]]:
    z = object_z()
    positions: dict[str, tuple[float, float]] = {}
    class_list = list(classes)

    if condition == "occlusion_25":
        positions[target_class] = (-0.12, -0.03)
        occluder = next(name for name in class_list if name != target_class)
        positions[occluder] = (-0.07, -0.09)
        remaining = [name for name in class_list if name not in positions]
        for idx, name in enumerate(remaining):
            positions[name] = (0.08 + idx * 0.13, 0.02 + idx * 0.08)
    elif condition == "occlusion_50":
        positions[target_class] = (-0.12, -0.02)
        occluder = next(name for name in class_list if name != target_class)
        positions[occluder] = (-0.10, -0.105)
        remaining = [name for name in class_list if name not in positions]
        for idx, name in enumerate(remaining):
            positions[name] = (0.08 + idx * 0.14, 0.04 + idx * 0.06)
    elif condition == "similar_distractor":
        if "amoxicillin" in class_list and "cefixime" in class_list:
            positions["amoxicillin"] = (-0.10, -0.04)
            positions["cefixime"] = (0.03, -0.02)
        if target_class not in positions:
            positions[target_class] = (-0.23, 0.01)
        remaining = [name for name in class_list if name not in positions]
        for idx, name in enumerate(remaining):
            positions[name] = (0.16 + idx * 0.12, 0.10)
    else:
        base_positions = [(-0.15, -0.04), (0.03, 0.04), (0.18, -0.10)]
        for name, xy in zip(class_list, base_positions):
            positions[name] = xy

    return {name: (x, y, z) for name, (x, y) in positions.items()}


def build_scene_metadata(
    condition: str,
    target_class: str,
    seed: int,
    use_textures: bool,
    resolution_width: int,
    resolution_height: int,
) -> dict:
    ensure_supported(condition, target_class)
    rng = random.Random(seed)
    classes = select_unique_classes(condition, target_class, rng)
    positions = condition_positions(condition, classes, target_class)
    yaw = yaw_for_condition(condition)
    objects = []

    for index, class_name in enumerate(classes):
        texture = find_front_texture(class_name) if use_textures else None
        object_yaw = yaw if class_name == target_class else rng.choice([0.0, 35.0, -35.0, 90.0])
        if condition.startswith("occlusion") and class_name != target_class and index == 1:
            object_yaw = yaw

        pose = {
            "position": [round(value, 4) for value in positions[class_name]],
            "rotation_euler_degrees": [0.0, 0.0, object_yaw],
        }
        objects.append(
            SceneObjectSpec(
                class_name=class_name,
                instance_id=f"{class_name}_{index:02d}",
                pose=pose,
                dimensions=dimensions_dict(),
                condition=condition,
                is_target=class_name == target_class,
                material_color=CLASS_COLORS[class_name],
                texture_path=str(texture.relative_to(project_root()).as_posix()) if texture else None,
                used_texture=bool(texture),
            ).to_metadata()
        )

    return {
        "condition": condition,
        "target_class": target_class,
        "seed": seed,
        "requested_use_textures": use_textures,
        "use_textures": any(obj["used_texture"] for obj in objects),
        "resolution": {
            "width": resolution_width,
            "height": resolution_height,
        },
        "objects": objects,
        "camera_pose": CAMERA_POSE,
        "table_pose": TABLE_POSE,
        "bin_pose": BIN_POSE,
    }


def expected_preview_paths(output_dir: Path) -> dict[str, tuple[Path, Path]]:
    return {
        condition: (
            output_dir / f"preview_{condition}.png",
            output_dir / f"preview_{condition}.json",
        )
        for condition in CONDITIONS
    }

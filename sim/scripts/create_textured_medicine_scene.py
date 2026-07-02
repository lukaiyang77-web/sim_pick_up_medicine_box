from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

from visual_scene_utils import (
    BIN_POSE,
    CAMERA_MODES,
    CAPTURE_MODES,
    DEBUG_BIG_CUBE,
    DEFAULT_OUTPUT_DIR,
    TABLE_POSE,
    bind_material_to_prim,
    build_scene_metadata,
    create_colored_material,
    create_textured_material,
    ensure_camera_mode,
    ensure_capture_mode,
    ensure_supported,
    project_root,
    resolve_project_path,
)


ISAAC_IMPORT_ERROR = "Isaac Sim modules not found. Please run this script with Isaac Sim python.sh."
TABLE_PRIM_PATH = "/World/Table"
BIN_PRIM_PATH = "/World/SortingBin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an Isaac Sim visual medicine-box scene.")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--target_class", required=True)
    parser.add_argument("--output_image", default=str(DEFAULT_OUTPUT_DIR / "preview.png"))
    parser.add_argument("--output_metadata", default=str(DEFAULT_OUTPUT_DIR / "preview.json"))
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_textures", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--camera_mode", choices=sorted(CAMERA_MODES), default="oblique")
    parser.add_argument("--capture_mode", choices=CAPTURE_MODES, default="camera_sensor")
    parser.add_argument("--box_scale", type=float, default=1.3)
    parser.add_argument("--debug_big_cube", action="store_true")
    parser.add_argument("--resolution_width", type=int, default=1024)
    parser.add_argument("--resolution_height", type=int, default=768)
    return parser.parse_args()


def start_simulation_app(headless: bool, width: int, height: int):
    try:
        try:
            from isaacsim import SimulationApp
        except ImportError:
            from omni.isaac.kit import SimulationApp
    except ImportError as exc:
        raise RuntimeError(ISAAC_IMPORT_ERROR) from exc

    return SimulationApp(
        {
            "headless": headless,
            "width": width,
            "height": height,
            "renderer": "RayTracedLighting",
        }
    )


def safe_import_runtime_modules():
    try:
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

        try:
            from isaacsim.core.api import World
        except ImportError:
            from omni.isaac.core import World
    except ImportError as exc:
        raise RuntimeError(ISAAC_IMPORT_ERROR) from exc

    return {
        "rep": rep,
        "omni_usd": omni.usd,
        "Gf": Gf,
        "Sdf": Sdf,
        "UsdGeom": UsdGeom,
        "UsdLux": UsdLux,
        "UsdShade": UsdShade,
        "World": World,
    }


def add_visible_cube(
    stage,
    UsdGeom,
    UsdShade,
    Sdf,
    Gf,
    prim_path: str,
    position: list[float],
    dimensions: list[float],
    color: list[float],
    yaw_degrees: float = 0.0,
    texture_path: Path | None = None,
):
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    prim = cube.GetPrim()
    xform = UsdGeom.XformCommonAPI(cube)
    xform.SetTranslate(tuple(position))
    xform.SetScale(tuple(dimensions))
    xform.SetRotate((0.0, 0.0, yaw_degrees), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
    material_name = prim_path.strip("/").replace("/", "_")
    texture_error = None
    texture_binding_method = "fallback_color"
    if texture_path is not None:
        try:
            material = create_textured_material(stage, UsdShade, Sdf, Gf, material_name, color, texture_path)
            texture_binding_method = "UsdShadePreviewSurface"
        except Exception as exc:
            texture_error = str(exc)
            print(f"Warning: textured material failed for {prim_path}: {texture_error}. Falling back to color.")
            material = create_colored_material(stage, UsdShade, Sdf, Gf, material_name, color)
    else:
        material = create_colored_material(stage, UsdShade, Sdf, Gf, material_name, color)
    material_bound = bind_material_to_prim(UsdShade, prim, material)
    return prim, material_bound, texture_binding_method, texture_error


def add_lighting(stage, UsdLux) -> None:
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(600.0)
    distant = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    distant.CreateIntensityAttr(1000.0)
    distant.CreateAngleAttr(0.5)


def camera_rotation_degrees(position: list[float], look_at: list[float]) -> tuple[float, float, float]:
    dx = look_at[0] - position[0]
    dy = look_at[1] - position[1]
    dz = look_at[2] - position[2]
    horizontal = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(horizontal, -dz))
    yaw = math.degrees(math.atan2(-dx, dy)) if horizontal > 1e-8 else 0.0
    return pitch, 0.0, yaw


def create_camera_prim(stage, UsdGeom, Gf, metadata: dict):
    camera_pose = metadata["camera_pose"]
    camera_path = metadata["camera_prim_path"]
    camera = UsdGeom.Camera.Define(stage, camera_path)
    prim = camera.GetPrim()
    camera.CreateFocalLengthAttr(camera_pose["focal_length"])
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    xform = UsdGeom.XformCommonAPI(camera)
    xform.SetTranslate(tuple(camera_pose["position"]))
    xform.SetRotate(
        camera_rotation_degrees(camera_pose["position"], camera_pose["look_at"]),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )
    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
    return prim


def vector_to_list(value) -> list[float] | None:
    if value is None:
        return None
    try:
        return [round(float(item), 5) for item in value]
    except TypeError:
        return [round(float(value), 5)]


def get_xform_values(UsdGeom, prim) -> tuple[list[float] | None, list[float] | None]:
    translation = None
    scale = None
    if not prim.IsValid():
        return translation, scale
    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        op_name = op.GetOpName()
        if op_name.endswith(":translate"):
            translation = vector_to_list(op.Get())
        elif op_name.endswith(":scale"):
            scale = vector_to_list(op.Get())
    return translation, scale


def inspect_prim(stage, UsdGeom, UsdShade, prim_path: str) -> dict:
    prim = stage.GetPrimAtPath(prim_path)
    info = {
        "prim_path": prim_path,
        "stage_prim_exists": prim.IsValid(),
        "prim_type_name": None,
        "actual_translation": None,
        "actual_scale": None,
        "visibility": None,
        "material_bound": False,
    }
    if not prim.IsValid():
        return info

    translation, scale = get_xform_values(UsdGeom, prim)
    info["prim_type_name"] = prim.GetTypeName()
    info["actual_translation"] = translation
    info["actual_scale"] = scale
    info["visibility"] = UsdGeom.Imageable(prim).ComputeVisibility()
    try:
        bound_material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        info["material_bound"] = bool(bound_material and bound_material.GetPrim().IsValid())
    except Exception:
        info["material_bound"] = False
    return info


def update_stage_metadata(stage, UsdGeom, UsdShade, metadata: dict) -> None:
    for obj in metadata["objects"]:
        prim_path = obj["prim_path"]
        obj.update(inspect_prim(stage, UsdGeom, UsdShade, prim_path))

    debug_cube = metadata.get("debug_big_cube", {})
    if debug_cube.get("enabled"):
        debug_cube.update(inspect_prim(stage, UsdGeom, UsdShade, debug_cube["prim_path"]))


def format_stage_dump(stage, UsdGeom, UsdShade, metadata: dict) -> str:
    prim_paths = [
        metadata["camera_prim_path"],
        TABLE_PRIM_PATH,
        BIN_PRIM_PATH,
        *[obj["prim_path"] for obj in metadata["objects"]],
        metadata["debug_big_cube"]["prim_path"],
    ]
    lines = [
        f"condition: {metadata['condition']}",
        f"target_class: {metadata['target_class']}",
        f"camera_mode: {metadata['camera_mode']}",
        f"capture_mode: {metadata['capture_mode']}",
        "",
    ]
    for prim_path in prim_paths:
        info = inspect_prim(stage, UsdGeom, UsdShade, prim_path)
        lines.append(f"prim_path: {prim_path}")
        lines.append(f"  exists: {info['stage_prim_exists']}")
        lines.append(f"  type: {info['prim_type_name']}")
        lines.append(f"  translation: {info['actual_translation']}")
        lines.append(f"  scale: {info['actual_scale']}")
        lines.append(f"  visibility: {info['visibility']}")
        lines.append(f"  material_bound: {info['material_bound']}")
    return "\n".join(lines) + "\n"


def write_stage_dump(stage, UsdGeom, UsdShade, metadata: dict, output_metadata: Path) -> None:
    if metadata["condition"] not in {"front", "similar_distractor"}:
        return
    dump_path = output_metadata.parent / f"stage_dump_{metadata['condition']}.txt"
    dump_path.write_text(format_stage_dump(stage, UsdGeom, UsdShade, metadata), encoding="utf-8")
    metadata["stage_dump_path"] = str(dump_path.relative_to(project_root()).as_posix())


def build_usd_scene(metadata: dict) -> dict:
    modules = safe_import_runtime_modules()
    rep = modules["rep"]
    omni_usd = modules["omni_usd"]
    Gf = modules["Gf"]
    Sdf = modules["Sdf"]
    UsdGeom = modules["UsdGeom"]
    UsdLux = modules["UsdLux"]
    UsdShade = modules["UsdShade"]
    World = modules["World"]

    world = World(stage_units_in_meters=1.0)
    stage = omni_usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdGeom.Scope.Define(stage, "/World/MedicineBoxes")

    add_lighting(stage, UsdLux)
    create_camera_prim(stage, UsdGeom, Gf, metadata)

    add_visible_cube(
        stage,
        UsdGeom,
        UsdShade,
        Sdf,
        Gf,
        TABLE_PRIM_PATH,
        TABLE_POSE["position"],
        TABLE_POSE["dimensions"],
        TABLE_POSE["color"],
    )
    add_visible_cube(
        stage,
        UsdGeom,
        UsdShade,
        Sdf,
        Gf,
        BIN_PRIM_PATH,
        BIN_POSE["position"],
        BIN_POSE["dimensions"],
        BIN_POSE["color"],
    )

    for obj in metadata["objects"]:
        dims = obj["dimensions"]
        dimensions = [dims["length"], dims["width"], dims["height"]]
        position = obj["pose"]["position"]
        yaw = obj["pose"]["rotation_euler_degrees"][2]
        prim_path = f"/World/MedicineBoxes/{obj['instance_id']}"
        obj["prim_path"] = prim_path
        texture_path = None
        if metadata["requested_use_textures"] and obj.get("texture_candidate_path"):
            texture_path = resolve_project_path(obj["texture_candidate_path"])
            if not texture_path.exists():
                obj["texture_error"] = f"texture candidate does not exist: {obj['texture_candidate_path']}"
                texture_path = None
        prim, material_bound, texture_binding_method, texture_error = add_visible_cube(
            stage,
            UsdGeom,
            UsdShade,
            Sdf,
            Gf,
            prim_path,
            position,
            dimensions,
            obj["material_color"],
            yaw,
            texture_path,
        )
        obj["used_texture"] = texture_binding_method != "fallback_color" and texture_error is None
        obj["texture_binding_method"] = texture_binding_method
        obj["texture_error"] = texture_error or obj.get("texture_error")
        if not obj["used_texture"] and obj["texture_error"] is None:
            obj["texture_error"] = (
                "use_textures disabled"
                if not metadata["requested_use_textures"]
                else "texture unavailable; used fallback color"
            )

    debug_cube = metadata.get("debug_big_cube", {})
    if debug_cube.get("enabled"):
        add_visible_cube(
            stage,
            UsdGeom,
            UsdShade,
            Sdf,
            Gf,
            debug_cube["prim_path"],
            debug_cube["position"],
            debug_cube["dimensions"],
            debug_cube["color"],
        )

    metadata["use_textures"] = any(obj.get("used_texture") for obj in metadata["objects"])

    world.reset()
    for _ in range(3):
        world.step(render=True)

    return {
        "rep": rep,
        "stage": stage,
        "world": world,
        "UsdGeom": UsdGeom,
        "UsdShade": UsdShade,
    }


def save_rgba_png(rgba, output_image: Path) -> None:
    output_image.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow and numpy are required to save camera sensor RGBA output.") from exc

    array = np.asarray(rgba)
    if array.size == 0:
        raise RuntimeError("Camera sensor returned an empty RGBA frame.")
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = (array * 255.0).clip(0, 255).astype(np.uint8)
        else:
            array = array.clip(0, 255).astype(np.uint8)
    if array.shape[-1] == 4:
        image = Image.fromarray(array, mode="RGBA")
    elif array.shape[-1] == 3:
        image = Image.fromarray(array, mode="RGB")
    else:
        raise RuntimeError(f"Unsupported camera frame shape: {array.shape}")
    image.save(output_image)


def capture_camera_sensor(world, metadata: dict, output_image: Path) -> None:
    try:
        try:
            from isaacsim.sensors.camera import Camera
        except ImportError:
            from omni.isaac.sensor import Camera
    except ImportError as exc:
        raise RuntimeError("Isaac Sim Camera sensor module is not available.") from exc

    camera = Camera(
        prim_path=metadata["camera_prim_path"],
        resolution=(metadata["resolution"]["width"], metadata["resolution"]["height"]),
    )
    camera.initialize()
    if hasattr(camera, "add_rgba_to_frame"):
        camera.add_rgba_to_frame()
    for _ in range(8):
        world.step(render=True)
    rgba = camera.get_rgba()
    save_rgba_png(rgba, output_image)


def copy_rendered_png(render_dir: Path, output_image: Path) -> None:
    pngs = sorted(render_dir.glob("**/*.png"), key=lambda path: path.stat().st_mtime)
    if not pngs:
        raise RuntimeError(f"No PNG was produced by the Replicator writer in {render_dir}.")
    output_image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pngs[-1], output_image)


def capture_replicator(rep, metadata: dict, output_image: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="visual_scene_v1_") as tmp:
        render_dir = Path(tmp)
        render_product = rep.create.render_product(
            metadata["camera_prim_path"],
            (metadata["resolution"]["width"], metadata["resolution"]["height"]),
        )
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(render_dir), rgb=True)
        writer.attach([render_product])
        rep.orchestrator.step()
        try:
            rep.orchestrator.wait_until_complete()
        except AttributeError:
            pass
        try:
            writer.detach()
        except AttributeError:
            pass
        copy_rendered_png(render_dir, output_image)


def capture_viewport(metadata: dict, output_image: Path) -> None:
    try:
        import omni.kit.viewport.utility as viewport_utility
    except ImportError as exc:
        raise RuntimeError("Viewport capture utility is not available.") from exc

    viewport = viewport_utility.get_active_viewport()
    if viewport is None:
        raise RuntimeError("No active viewport is available for viewport capture.")
    viewport.camera_path = metadata["camera_prim_path"]
    output_image.parent.mkdir(parents=True, exist_ok=True)
    capture = viewport_utility.capture_viewport_to_file(viewport, str(output_image))
    if hasattr(capture, "wait_for_result"):
        capture.wait_for_result()
    if not output_image.exists() or output_image.stat().st_size <= 0:
        raise RuntimeError(f"Viewport capture did not create a valid PNG: {output_image}")


def capture_scene(runtime: dict, metadata: dict, output_image: Path) -> None:
    requested = metadata["requested_capture_mode"]
    order = [requested, *[mode for mode in CAPTURE_MODES if mode != requested]]
    errors: list[str] = []
    for mode in order:
        try:
            if mode == "camera_sensor":
                capture_camera_sensor(runtime["world"], metadata, output_image)
            elif mode == "replicator":
                capture_replicator(runtime["rep"], metadata, output_image)
            elif mode == "viewport":
                capture_viewport(metadata, output_image)
            else:
                raise RuntimeError(f"Unknown capture mode: {mode}")
            metadata["capture_mode"] = mode
            metadata["actual_capture_camera_path"] = metadata["camera_prim_path"]
            if errors:
                metadata["capture_fallback_warnings"] = errors
            return
        except Exception as exc:
            message = f"{mode} capture failed: {exc}"
            errors.append(message)
            print(f"Warning: {message}")
    raise RuntimeError("All capture modes failed. " + " | ".join(errors))


def log_scene_debug(metadata: dict) -> None:
    camera_pose = metadata["camera_pose"]
    print(f"Scene condition: {metadata['condition']}")
    print(f"Target class: {metadata['target_class']}")
    print(f"Camera mode: {metadata['camera_mode']}")
    print(f"Requested capture mode: {metadata['requested_capture_mode']}")
    print(f"Camera prim path: {metadata['camera_prim_path']}")
    print(f"Camera position: {camera_pose['position']}")
    print(f"Camera look_at: {camera_pose['look_at']}")
    for obj in metadata["objects"]:
        prim_path = f"/World/MedicineBoxes/{obj['instance_id']}"
        print(
            "Object: "
            f"class_name={obj['class_name']} "
            f"position={obj['pose']['position']} "
            f"dimensions={obj['dimensions']} "
            f"prim_path={prim_path}"
        )
    debug_cube = metadata.get("debug_big_cube", {})
    if debug_cube.get("enabled"):
        print(
            "DebugBigRedCube: "
            f"position={debug_cube['position']} "
            f"dimensions={debug_cube['dimensions']} "
            f"prim_path={debug_cube['prim_path']}"
        )


def write_metadata(metadata: dict, output_metadata: Path) -> None:
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        ensure_supported(args.condition, args.target_class)
        ensure_camera_mode(args.camera_mode)
        ensure_capture_mode(args.capture_mode)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.box_scale <= 0:
        print(f"box_scale must be positive, got {args.box_scale}.", file=sys.stderr)
        return 2

    output_image = resolve_project_path(args.output_image)
    output_metadata = resolve_project_path(args.output_metadata)
    metadata = build_scene_metadata(
        condition=args.condition,
        target_class=args.target_class,
        seed=args.seed,
        use_textures=args.use_textures,
        resolution_width=args.resolution_width,
        resolution_height=args.resolution_height,
        camera_mode=args.camera_mode,
        box_scale=args.box_scale,
        capture_mode=args.capture_mode,
        debug_big_cube=args.debug_big_cube,
    )

    app = None
    try:
        log_scene_debug(metadata)
        app = start_simulation_app(args.headless, args.resolution_width, args.resolution_height)
        runtime = build_usd_scene(metadata)
        update_stage_metadata(runtime["stage"], runtime["UsdGeom"], runtime["UsdShade"], metadata)
        write_stage_dump(runtime["stage"], runtime["UsdGeom"], runtime["UsdShade"], metadata, output_metadata)
        capture_scene(runtime, metadata, output_image)
        metadata["image_path"] = str(output_image.relative_to(project_root()).as_posix())
        write_metadata(metadata, output_metadata)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    finally:
        if app is not None:
            app.close()

    print(f"Wrote image: {output_image}")
    print(f"Wrote metadata: {output_metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

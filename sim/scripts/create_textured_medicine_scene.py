from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from visual_scene_utils import (
    BIN_POSE,
    DEFAULT_OUTPUT_DIR,
    TABLE_POSE,
    build_scene_metadata,
    ensure_supported,
    project_root,
    resolve_project_path,
)


ISAAC_IMPORT_ERROR = "Isaac Sim modules not found. Please run this script with Isaac Sim python.sh."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a textured Isaac Sim medicine-box scene.")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--target_class", required=True)
    parser.add_argument("--output_image", default=str(DEFAULT_OUTPUT_DIR / "preview.png"))
    parser.add_argument("--output_metadata", default=str(DEFAULT_OUTPUT_DIR / "preview.json"))
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_textures", action=argparse.BooleanOptionalAction, default=True)
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

    return rep, omni.usd, Gf, Sdf, UsdGeom, UsdLux, UsdShade, World


def create_material(stage, UsdShade, Sdf, Gf, name: str, color: list[float], texture_path: Path | None):
    material_path = f"/World/Looks/{name}"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    diffuse_input.Set(Gf.Vec3f(*color))

    if texture_path is not None:
        texture = UsdShade.Shader.Define(stage, f"{material_path}/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture_path.as_posix()))
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        diffuse_input.ConnectToSource(texture.ConnectableAPI(), "rgb")

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_material(UsdShade, prim, material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim)
    UsdShade.MaterialBindingAPI(prim).Bind(material)


def add_cube(UsdGeom, stage, prim_path: str, position: list[float], dimensions: list[float], yaw_degrees: float):
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(cube)
    xform.SetTranslate(tuple(position))
    xform.SetScale(tuple(dimensions))
    xform.SetRotate((0.0, 0.0, yaw_degrees), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    return cube.GetPrim()


def add_text_label(stage, UsdGeom, Gf, text: str, position: list[float], yaw_degrees: float):
    text_path = f"/World/Labels/{text}"
    text_prim = UsdGeom.Text.Define(stage, text_path)
    text_prim.CreateTextAttr(text)
    text_prim.CreateHeightAttr(0.018)
    text_prim.CreateAlignAttr("center")
    text_prim.CreateAxisAttr("Z")
    xform = UsdGeom.XformCommonAPI(text_prim)
    xform.SetTranslate((position[0], position[1], position[2] + 0.019))
    xform.SetRotate((0.0, 0.0, yaw_degrees), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    xform.SetScale((1.0, 1.0, 1.0))
    return text_prim.GetPrim()


def add_lighting(stage, UsdLux) -> None:
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(450.0)
    distant = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    distant.CreateIntensityAttr(900.0)
    distant.CreateAngleAttr(0.4)


def build_usd_scene(metadata: dict) -> tuple[object, object]:
    rep, omni_usd, Gf, Sdf, UsdGeom, UsdLux, UsdShade, World = safe_import_runtime_modules()
    world = World(stage_units_in_meters=1.0)
    stage = omni_usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdGeom.Scope.Define(stage, "/World/Labels")

    add_lighting(stage, UsdLux)

    table_material = create_material(stage, UsdShade, Sdf, Gf, "table_mat", [0.70, 0.70, 0.68], None)
    table_prim = add_cube(
        UsdGeom,
        stage,
        "/World/Table",
        TABLE_POSE["position"],
        TABLE_POSE["dimensions"],
        0.0,
    )
    bind_material(UsdShade, table_prim, table_material)

    bin_material = create_material(stage, UsdShade, Sdf, Gf, "bin_mat", BIN_POSE["color"], None)
    bin_prim = add_cube(
        UsdGeom,
        stage,
        "/World/SortingBin",
        BIN_POSE["position"],
        BIN_POSE["dimensions"],
        0.0,
    )
    bind_material(UsdShade, bin_prim, bin_material)

    for obj in metadata["objects"]:
        color = obj["material_color"]
        texture = resolve_project_path(obj["texture_path"]) if obj["texture_path"] else None
        if texture is not None and not texture.exists():
            print(f"Warning: texture missing for {obj['class_name']}: {texture}. Falling back to color.")
            texture = None
            obj["used_texture"] = False

        if obj["texture_path"] is None and metadata["requested_use_textures"]:
            print(f"Warning: no texture found for {obj['class_name']}. Falling back to color.")

        try:
            material = create_material(
                stage,
                UsdShade,
                Sdf,
                Gf,
                f"{obj['instance_id']}_mat",
                color,
                texture,
            )
        except Exception as exc:
            print(
                f"Warning: texture material failed for {obj['class_name']}: {exc}. "
                "Falling back to color."
            )
            obj["used_texture"] = False
            material = create_material(
                stage,
                UsdShade,
                Sdf,
                Gf,
                f"{obj['instance_id']}_fallback_mat",
                color,
                None,
            )
        dims = obj["dimensions"]
        dimensions = [dims["length"], dims["width"], dims["height"]]
        position = obj["pose"]["position"]
        yaw = obj["pose"]["rotation_euler_degrees"][2]
        prim = add_cube(
            UsdGeom,
            stage,
            f"/World/MedicineBoxes/{obj['instance_id']}",
            position,
            dimensions,
            yaw,
        )
        bind_material(UsdShade, prim, material)
        try:
            add_text_label(stage, UsdGeom, Gf, obj["class_name"], position, yaw)
        except Exception as exc:
            print(f"Warning: label creation failed for {obj['class_name']}: {exc}.")

    metadata["use_textures"] = any(obj["used_texture"] for obj in metadata["objects"])

    camera_pose = metadata["camera_pose"]
    camera = rep.create.camera(
        position=tuple(camera_pose["position"]),
        look_at=tuple(camera_pose["look_at"]),
        focal_length=24.0,
    )
    world.reset()
    return rep, camera


def copy_rendered_png(render_dir: Path, output_image: Path) -> None:
    pngs = sorted(render_dir.glob("**/*.png"), key=lambda path: path.stat().st_mtime)
    if not pngs:
        raise RuntimeError(f"No PNG was produced by the Replicator writer in {render_dir}.")
    output_image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pngs[-1], output_image)


def render_scene(metadata: dict, output_image: Path) -> None:
    rep, camera = build_usd_scene(metadata)
    with tempfile.TemporaryDirectory(prefix="visual_scene_v1_") as tmp:
        render_dir = Path(tmp)
        render_product = rep.create.render_product(
            camera,
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


def write_metadata(metadata: dict, output_metadata: Path) -> None:
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        ensure_supported(args.condition, args.target_class)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
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
    )

    if args.use_textures and not metadata["use_textures"]:
        print("Warning: no front textures found. Falling back to solid colors and 3D text labels.")

    app = None
    try:
        app = start_simulation_app(args.headless, args.resolution_width, args.resolution_height)
        render_scene(metadata, output_image)
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

# Isaac Sim visual medicine scene v1

This module builds a lightweight Isaac Sim visualization scene for the medicine-box sorting project. It is meant for midterm demos and simulation validation: a tabletop, a sorting bin, a fixed RGB camera, and 3 to 5 cuboid medicine boxes with either front textures or solid-color fallback materials.

The current scope is visualization only. It does not train perception models, connect CLIP/DINOv2/linear probes, use OpenVLA or pi0, implement robot-arm IK, or simulate a physical grasp. Existing perception experiments and L0/L1 oracle baseline outputs are not required by these scripts and should remain untouched.

## Relationship to L0/L1 baselines

The L0 and L1 baselines validate simplified sorting behavior. Visual scene v1 is a separate rendering layer that creates reproducible preview images and JSON metadata for later demo and perception/selector integration. It does not replace or overwrite baseline results.

## Files

- `sim/scripts/create_textured_medicine_scene.py`: generate one condition preview and metadata file.
- `sim/scripts/generate_visual_scene_previews.py`: generate all six condition previews.
- `sim/scripts/check_visual_scene_outputs.py`: validate generated PNG and JSON outputs.
- `sim/scripts/visual_scene_utils.py`: shared classes, scene layout, paths, and metadata helpers.
- `sim/results/visual_scene_v1/`: default output directory. Generated files are ignored by Git.

## Conditions

The supported conditions are:

- `front`
- `back`
- `side`
- `occlusion_25`
- `occlusion_50`
- `similar_distractor`

Medicine classes are fixed to:

- `amoxicillin`
- `cefixime`
- `ibuprofen`
- `montmorillonite`
- `vitamin_c`

Each medicine box is represented as a cuboid with dimensions `0.12m x 0.07m x 0.035m`.

## Run one scene

Run from the project root on the Ubuntu + Isaac Sim machine:

```bash
/path/to/isaac-sim/python.sh sim/scripts/create_textured_medicine_scene.py \
  --condition similar_distractor \
  --target_class amoxicillin \
  --output_image sim/results/visual_scene_v1/preview_similar_distractor.png \
  --output_metadata sim/results/visual_scene_v1/preview_similar_distractor.json \
  --headless \
  --seed 42 \
  --use_textures
```

Do not hard-code the Isaac Sim installation path in the scripts. Replace `/path/to/isaac-sim/python.sh` with the real path only in the shell command used on the 5080 machine.

## Generate all previews

```bash
/path/to/isaac-sim/python.sh sim/scripts/generate_visual_scene_previews.py \
  --output_dir sim/results/visual_scene_v1 \
  --headless \
  --use_textures \
  --seed 42
```

Expected outputs:

- `sim/results/visual_scene_v1/preview_front.png`
- `sim/results/visual_scene_v1/preview_front.json`
- `sim/results/visual_scene_v1/preview_back.png`
- `sim/results/visual_scene_v1/preview_back.json`
- `sim/results/visual_scene_v1/preview_side.png`
- `sim/results/visual_scene_v1/preview_side.json`
- `sim/results/visual_scene_v1/preview_occlusion_25.png`
- `sim/results/visual_scene_v1/preview_occlusion_25.json`
- `sim/results/visual_scene_v1/preview_occlusion_50.png`
- `sim/results/visual_scene_v1/preview_occlusion_50.json`
- `sim/results/visual_scene_v1/preview_similar_distractor.png`
- `sim/results/visual_scene_v1/preview_similar_distractor.json`

## Check outputs

```bash
python3 sim/scripts/check_visual_scene_outputs.py \
  --output_dir sim/results/visual_scene_v1
```

The checker writes:

```text
sim/results/visual_scene_v1/check_summary.txt
```

It verifies that all six PNG and JSON files exist, PNG files are non-empty, required metadata fields are present, each scene has 3 to 5 medicine boxes, the target appears exactly once, class names are not duplicated, and `similar_distractor` contains both `amoxicillin` and `cefixime`.

## Texture behavior

The scene generator looks for front images under:

```text
data/textures/medicine_box_images_v3/
```

For each class it tries paths such as:

```text
views/front.png
raw/front_box.png
raw/front_blue.png
raw/front_red.png
```

If no texture is found, or if a texture is unavailable at render time, the script prints a warning and falls back to a distinct solid color plus a 3D text label. Metadata records `requested_use_textures`, top-level `use_textures`, and each object's `texture_path` and `used_texture`.

## Python choice

普通 `python3` 可以运行纯 Python 检查脚本，也可以做语法检查：

```bash
python3 -m py_compile sim/scripts/create_textured_medicine_scene.py
python3 -m py_compile sim/scripts/generate_visual_scene_previews.py
python3 -m py_compile sim/scripts/check_visual_scene_outputs.py
```

Rendering scripts need Isaac Sim modules, so actual image generation should use Isaac Sim's bundled `python.sh`. If the scripts are run with ordinary Python, they will stop with:

```text
Isaac Sim modules not found. Please run this script with Isaac Sim python.sh.
```

## GPU rendering troubleshooting

If rendering fails on the 5080 machine:

- Confirm the command uses Isaac Sim's `python.sh`, not system Python.
- Try `--headless` first, then `--no-headless` if an interactive viewport is needed.
- Check that the NVIDIA driver and Isaac Sim version are compatible.
- Reduce resolution with `--resolution_width 640 --resolution_height 480` to isolate memory or renderer issues.
- Confirm the output directory is writable.
- Re-run with `--no-use_textures` to separate texture/material issues from scene construction.

## Current limitations

- Medicine boxes are simplified cuboids.
- Texture mapping is best-effort and may fall back to solid colors.
- The first version does not implement real robot-arm grasping.
- The first version does not connect to VLA models.
- The goal is a reproducible, demo-ready visual simulation scene for screenshots and metadata.

## Next steps

- Connect the L1 stable baseline to the visual scene.
- Connect a perception selector.
- Add real medicine-box USD assets or higher-quality texture placement.
- Add GPU PhysX validation.

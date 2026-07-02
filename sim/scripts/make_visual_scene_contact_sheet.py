from __future__ import annotations

import argparse
from pathlib import Path

from visual_scene_utils import CONDITIONS, DEFAULT_OUTPUT_DIR, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a 2x3 contact sheet for visual scene v1 previews.")
    parser.add_argument("--input_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--output_path",
        default=str(DEFAULT_OUTPUT_DIR / "contact_sheet_visual_scene_v1.png"),
    )
    parser.add_argument("--thumb_width", type=int, default=512)
    parser.add_argument("--thumb_height", type=int, default=384)
    return parser.parse_args()


def load_font(size: int):
    from PIL import ImageFont

    for font_name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def resize_letterboxed(image, size: tuple[int, int]):
    from PIL import Image, ImageOps

    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return ImageOps.expand(canvas, border=1, fill=(220, 220, 220))


def make_contact_sheet(input_dir: Path, output_path: Path, thumb_width: int, thumb_height: int) -> None:
    from PIL import Image, ImageDraw

    if thumb_width <= 0 or thumb_height <= 0:
        raise ValueError("thumb_width and thumb_height must be positive.")

    missing = [
        input_dir / f"preview_{condition}.png"
        for condition in CONDITIONS
        if not (input_dir / f"preview_{condition}.png").exists()
    ]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing preview PNG files:\n{missing_text}")

    title_height = 36
    padding = 16
    cell_width = thumb_width + 2
    cell_height = title_height + thumb_height + 2
    sheet_width = padding * 4 + cell_width * 3
    sheet_height = padding * 3 + cell_height * 2
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_font(22)

    for index, condition in enumerate(CONDITIONS):
        row = index // 3
        col = index % 3
        x = padding + col * (cell_width + padding)
        y = padding + row * (cell_height + padding)
        text_bbox = draw.textbbox((0, 0), condition, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        draw.text((x + (cell_width - text_width) // 2, y), condition, fill="black", font=font)

        image = Image.open(input_dir / f"preview_{condition}.png")
        thumb = resize_letterboxed(image, (thumb_width, thumb_height))
        sheet.paste(thumb, (x, y + title_height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    print(f"Wrote contact sheet: {output_path}")


def main() -> int:
    args = parse_args()
    make_contact_sheet(
        resolve_project_path(args.input_dir),
        resolve_project_path(args.output_path),
        args.thumb_width,
        args.thumb_height,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

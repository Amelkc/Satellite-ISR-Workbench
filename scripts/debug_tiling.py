from __future__ import annotations

from pathlib import Path
import argparse

import cv2
import numpy as np
import yaml

from src.preprocessing.tile import compute_tile_grid, TileSpec


def draw_tile_grid(
    image: np.ndarray,
    tiles: list[TileSpec],
    tile_size: int,
    overlap: int,
    line_thickness: int = 3,
) -> np.ndarray:
    canvas = image.copy()
    stride = tile_size - overlap

    for idx, tile in enumerate(tiles):
        # Full tile boundary in green.
        cv2.rectangle(
            canvas,
            (tile.x0, tile.y0),
            (tile.x1, tile.y1),
            color=(0, 255, 0),
            thickness=line_thickness,
        )

        # Overlap zone (this tile's leading overlap with previous tile) in red,
        # only drawn where an actual overlap exists.
        if tile.x0 + overlap < tile.x1 and tile.x0 > 0:
            cv2.rectangle(
                canvas,
                (tile.x0, tile.y0),
                (tile.x0 + overlap, tile.y1),
                color=(0, 0, 255),
                thickness=-1,
            )
        if tile.y0 + overlap < tile.y1 and tile.y0 > 0:
            cv2.rectangle(
                canvas,
                (tile.x0, tile.y0),
                (tile.x1, tile.y0 + overlap),
                color=(0, 0, 255),
                thickness=-1,
            )

        # Tile index label.
        label = f"{idx}"
        cv2.putText(
            canvas,
            label,
            (tile.x0 + 8, tile.y0 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Blend the solid red overlap fills with the original image for transparency.
    overlay = cv2.addWeighted(canvas, 0.35, image, 0.65, 0)

    # Re-draw crisp green tile borders and labels on top of the blended overlay
    # so boundaries stay sharp while overlap fill stays semi-transparent.
    for idx, tile in enumerate(tiles):
        cv2.rectangle(
            overlay,
            (tile.x0, tile.y0),
            (tile.x1, tile.y1),
            color=(0, 255, 0),
            thickness=line_thickness,
        )
        cv2.putText(
            overlay,
            f"{idx}",
            (tile.x0 + 8, tile.y0 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return overlay


def add_legend(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    legend_h = 60
    canvas = np.full((h + legend_h, w, 3), 255, dtype=np.uint8)
    canvas[:h] = image

    cv2.rectangle(canvas, (10, h + 15), (40, h + 35), (0, 255, 0), 3)
    cv2.putText(canvas, "tile boundary", (50, h + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (260, h + 15), (290, h + 35), (0, 0, 255), -1)
    cv2.putText(canvas, "overlap region", (300, h + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Visualize the tile grid before running full tiling.")
    parser.add_argument("--image", type=str, required=True, help="Path to a full-size DOTA image.")
    parser.add_argument("--config", type=str, default="configs/tiling.yaml", help="Path to tiling config.")
    parser.add_argument("--out", type=str, default=None, help="Path to save the debug visualization.")
    parser.add_argument("--max-preview-size", type=int, default=1600,
                         help="Resize the longest side of the preview for easier viewing.")
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tile_size = config["tiling"]["tile_size"]
    overlap = config["tiling"]["overlap"]

    h, w = image.shape[:2]
    tiles = compute_tile_grid(w, h, tile_size, overlap)

    print(f"Image: {image_path.name} ({w}x{h})")
    print(f"tile_size={tile_size}, overlap={overlap} -> {len(tiles)} tiles")

    debug_img = draw_tile_grid(image, tiles, tile_size, overlap)
    debug_img = add_legend(debug_img)

    long_side = max(debug_img.shape[:2])
    if long_side > args.max_preview_size:
        scale = args.max_preview_size / long_side
        debug_img = cv2.resize(debug_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    out_path = Path(args.out) if args.out else Path("assets/figures/tiling_debug") / f"{image_path.stem}_grid.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), debug_img)

    print(f"Saved grid preview to {out_path}")


if __name__ == "__main__":
    main()
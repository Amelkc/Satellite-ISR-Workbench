from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import argparse

import cv2
import numpy as np
import yaml
from shapely.geometry import Polygon, box

from src.preprocessing.parse_dota import DotaObj, parse_dota_label_file


@dataclass
class TileSpec:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def load_tiling_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_tile_grid(
    image_w: int,
    image_h: int,
    tile_size: int,
    overlap: int,
) -> List[TileSpec]:
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")

    tiles: List[TileSpec] = []

    y0 = 0
    while y0 < image_h:
        y1 = min(y0 + tile_size, image_h)
        x0 = 0
        while x0 < image_w:
            x1 = min(x0 + tile_size, image_w)
            tiles.append(TileSpec(x0=x0, y0=y0, x1=x1, y1=y1))

            if x1 == image_w:
                break
            x0 += stride

        if y1 == image_h:
            break
        y0 += stride

    return tiles


def polygon_to_shapely(polygon: List[List[float]]) -> Polygon:
    return Polygon(polygon)


def clip_polygon_to_tile(
    obj_polygon: List[List[float]],
    tile: TileSpec,
) -> Tuple[List[List[float]] | None, float]:
    """
    Clips a DOTA polygon against a tile's bounding box.

    Returns:
        (clipped_polygon_in_tile_coords, visible_ratio)
        clipped_polygon_in_tile_coords is None if there is no intersection.
    """
    original_poly = polygon_to_shapely(obj_polygon)
    if not original_poly.is_valid or original_poly.area == 0:
        return None, 0.0

    tile_box = box(tile.x0, tile.y0, tile.x1, tile.y1)
    intersection = original_poly.intersection(tile_box)

    if intersection.is_empty:
        return None, 0.0

    visible_ratio = intersection.area / original_poly.area

    # Intersection may be a Polygon, MultiPolygon, or degenerate geometry.
    if intersection.geom_type == "Polygon":
        coords = list(intersection.exterior.coords)[:-1]  # drop closing duplicate
    elif intersection.geom_type == "MultiPolygon":
        largest = max(intersection.geoms, key=lambda g: g.area)
        coords = list(largest.exterior.coords)[:-1]
    else:
        return None, 0.0

    # Remap to tile-local coordinates.
    local_coords = [[x - tile.x0, y - tile.y0] for x, y in coords]

    return local_coords, visible_ratio


def polygon_to_min_area_rect_points(polygon: List[List[float]]) -> List[List[float]]:
    """
    Converts an arbitrary clipped polygon back into a 4-point oriented box
    using OpenCV's minAreaRect, so downstream OBB tooling always receives
    exactly 4 vertices per object.
    """
    pts = np.array(polygon, dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    box_pts = cv2.boxPoints(rect)
    return box_pts.tolist()


def tile_objects_for_tile(
    objects: List[DotaObj],
    tile: TileSpec,
    min_object_area_px: float,
    min_visible_ratio: float,
) -> List[DotaObj]:
    kept: List[DotaObj] = []

    for obj in objects:
        clipped, visible_ratio = clip_polygon_to_tile(obj.polygon, tile)

        if clipped is None:
            continue
        if visible_ratio < min_visible_ratio:
            continue

        clipped_poly = Polygon(clipped)
        if not clipped_poly.is_valid or clipped_poly.area < min_object_area_px:
            continue

        obb_points = polygon_to_min_area_rect_points(clipped)

        kept.append(
            DotaObj(
                image_id=f"{obj.image_id}",
                category=obj.category,
                difficult=obj.difficult,
                polygon=obb_points,
            )
        )

    return kept


def write_dota_label_file(objects: List[DotaObj], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for obj in objects:
            flat = []
            for x, y in obj.polygon:
                flat.extend([f"{x:.2f}", f"{y:.2f}"])
            line = " ".join(flat) + f" {obj.category} {obj.difficult}\n"
            f.write(line)


def tile_single_image(
    image_path: Path,
    label_path: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    tile_size: int,
    overlap: int,
    min_object_area_px: float,
    min_visible_ratio: float,
    drop_empty_tiles: bool,
) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skipping unreadable image: {image_path}")
        return 0

    h, w = image.shape[:2]
    objects = parse_dota_label_file(label_path) if label_path.exists() else []

    tiles = compute_tile_grid(w, h, tile_size, overlap)
    saved_count = 0

    for idx, tile in enumerate(tiles):
        tile_objects = tile_objects_for_tile(
            objects,
            tile,
            min_object_area_px=min_object_area_px,
            min_visible_ratio=min_visible_ratio,
        )

        if drop_empty_tiles and not tile_objects:
            continue

        tile_img = image[tile.y0:tile.y1, tile.x0:tile.x1]
        tile_name = f"{image_path.stem}_tile{idx:03d}"

        out_img_path = out_images_dir / f"{tile_name}.png"
        out_lbl_path = out_labels_dir / f"{tile_name}.txt"

        out_images_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img_path), tile_img)
        write_dota_label_file(tile_objects, out_lbl_path)

        saved_count += 1

    return saved_count


def tile_dataset(config: dict) -> None:
    paths = config["paths"]
    tiling_cfg = config["tiling"]
    image_cfg = config["image"]

    images_dir = Path(paths["images_dir"])
    labels_dir = Path(paths["labels_dir"])
    out_images_dir = Path(paths["out_images_dir"])
    out_labels_dir = Path(paths["out_labels_dir"])

    extensions = image_cfg["extensions"]
    image_files = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in extensions
    )

    total_tiles = 0
    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"

        saved = tile_single_image(
            image_path=image_path,
            label_path=label_path,
            out_images_dir=out_images_dir,
            out_labels_dir=out_labels_dir,
            tile_size=tiling_cfg["tile_size"],
            overlap=tiling_cfg["overlap"],
            min_object_area_px=tiling_cfg["min_object_area_px"],
            min_visible_ratio=tiling_cfg["min_visible_ratio"],
            drop_empty_tiles=tiling_cfg["drop_empty_tiles"],
        )

        total_tiles += saved
        print(f"{image_path.name}: saved {saved} tiles")

    print(f"Done. Total tiles saved: {total_tiles}")


def main():
    parser = argparse.ArgumentParser(description="Tile DOTA images and remap OBB labels.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tiling.yaml",
        help="Path to tiling YAML config.",
    )
    args = parser.parse_args()

    config = load_tiling_config(args.config)
    tile_dataset(config)


if __name__ == "__main__":
    main()
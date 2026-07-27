from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import random
import cv2
import numpy as np
import json, os
from dotenv import load_dotenv

from src.preprocessing.parse_dota import DotaObj, parse_dota_label_file
load_dotenv()

LABEL_DIR=os.getenv('ANOT_TRAIN')
IMG_DIR=os.getenv('TRAIN_DIR')

DEFAULT_COLORS = {
    "plane": (0, 255, 255),
    "helicopter": (255, 0, 255),
    "ship": (255, 255, 0),
    "small-vehicle": (0, 255, 0),
    "large-vehicle": (0, 165, 255),
}


def get_color(category: str) -> tuple[int, int, int]:
    if category in DEFAULT_COLORS:
        return DEFAULT_COLORS[category]
    rng = random.Random(category)
    return (
        rng.randint(80, 255),
        rng.randint(80, 255),
        rng.randint(80, 255),
    )


def draw_polygon(
    image: np.ndarray,
    polygon: List[List[float]],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
    return image


def draw_label(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color: tuple[int, int, int],
) -> np.ndarray:
    x, y = xy
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(image, (x, max(0, y - th - 8)), (x + tw + 6, y + 4), color, -1)
    cv2.putText(
        image,
        text,
        (x + 3, y - 4),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    return image


def annotate_image(
    image: np.ndarray,
    objects: List[DotaObj],
    draw_difficult: bool = False,
) -> np.ndarray:
    canvas = image.copy()

    for obj in objects:
        color = get_color(obj.category)
        canvas = draw_polygon(canvas, obj.polygon, color=color, thickness=2)

        x0 = int(min(p[0] for p in obj.polygon))
        y0 = int(min(p[1] for p in obj.polygon))
        label = obj.category
        if draw_difficult:
            label += f" | diff={obj.difficult}"

        canvas = draw_label(canvas, label, (x0, y0), color=color)

    return canvas


def visualize_single(
    image_path: str | Path,
    label_path: str | Path,
    out_path: str | Path | None = None,
) -> np.ndarray:
    image_path = Path(image_path)
    label_path = Path(label_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    objects = parse_dota_label_file(label_path)
    annotated = annotate_image(image, objects)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)

    return annotated


def visualize_folder(
    images_dir: str | Path,
    labels_dir: str | Path,
    out_dir: str | Path,
    limit: int = 20,
) -> None:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(images_dir.glob("[!._]*.png")) + list(images_dir.glob("[!._]*.jpg")))

    count = 0
    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        out_path = out_dir / f"{image_path.stem}.png"
        visualize_single(image_path, label_path, out_path)
        count += 1

        if count >= limit:
            break

    print(f"Saved {count} annotated samples to {out_dir}")


if __name__ == "__main__":
    visualize_folder(
        images_dir=IMG_DIR,
        labels_dir=LABEL_DIR,
        out_dir="assets/figures/dota_samples",
        limit=5,
    )
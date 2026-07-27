from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any
import json, os
from dotenv import load_dotenv

# This module parses a single DOTA label file or an entire folder of label files into structured Python dictionaries.
# The template matches the official DOTA line format of 8 coordinates + category + difficult flag.
load_dotenv()

LABEL_DIR=os.getenv('ANOT_TRAIN')


@dataclass
class DotaObj:
    image_id: str
    category: str
    difficult: int
    polygon: List[List[float]]
    
    @property
    def xs(self):
        return [pt[0] for pt in self.polygon]
    
    @property
    def ys(self):
        return [pt[1] for pt in self.polygon]
    
    @property
    def bbox_minmax(self) -> List[float]:
        return [min(self.xs), min(self.ys), max(self.xs), max(self.ys)]

    @property
    def center(self) -> List[float]:
        return [sum(self.xs) / 4.0, sum(self.ys) / 4.0]
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["bbox_minmax"] = self.bbox_minmax
        d["center"] = self.center
        return d
    
def parse_dota_line(line: str, image_id: str) -> DotaObj | None:
    line = line.strip()
    if not line:
        return None

    cols = line.split()
    if len(cols) < 10:
        return None

    coords = list(map(float, cols[:8]))
    category = cols[8]
    difficult = int(cols[9])

    polygon = [
        [coords[0], coords[1]],
        [coords[2], coords[3]],
        [coords[4], coords[5]],
        [coords[6], coords[7]],
    ]

    return DotaObj(
        image_id=image_id,
        category=category,
        difficult=difficult,
        polygon=polygon,
    )
    
def parse_dota_label_file(label_path: str | Path) -> List[DotaObj]:
    label_path = Path(label_path)
    image_id = label_path.stem
    objects: List[DotaObj] = []

    with label_path.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            if line.lower().startswith("imagesource"):
                continue
            if line.lower().startswith("gsd"):
                continue

            obj = parse_dota_line(line, image_id=image_id)
            if obj is not None:
                objects.append(obj)

    return objects

def parse_dota_label_dir(label_dir: str | Path) -> Dict[str, List[DotaObj]]:
    label_dir = Path(label_dir)
    parsed: Dict[str, List[DotaObj]] = {}

    for txt_file in sorted(label_dir.glob("[!._]*.txt")): #avoid macos hidden files
        print(txt_file)
        parsed[txt_file.stem] = parse_dota_label_file(txt_file)

    return parsed


def export_parsed_labels_to_json(label_dir: str | Path, out_json: str | Path) -> None:
    parsed = parse_dota_label_dir(label_dir)
    serializable = {
        image_id: [obj.to_dict() for obj in objects]
        for image_id, objects in parsed.items()
    }

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


if __name__ == "__main__":
    label_dir = Path(LABEL_DIR)
    out_json = Path("data/parsed_labels_train.json")

    if label_dir.exists():
        export_parsed_labels_to_json(label_dir, out_json)
        print(f"Saved parsed annotations to {out_json}")
    else:
        print(f"Label directory not found: {label_dir}")

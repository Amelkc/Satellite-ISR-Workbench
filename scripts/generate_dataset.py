import random
import shutil
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
IMAGE_DIR = Path(os.getenv('IMG'))
ANNOTATION_DIR = Path(os.getenv('ANNOT'))
OUTPUT_DIR = Path(os.getenv('OUT'))

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42

output_dirs = {
    "train_img": OUTPUT_DIR / "train_img",
    "val_img": OUTPUT_DIR / "val_img",
    "test_img": OUTPUT_DIR / "test_img",
    "train_annot": OUTPUT_DIR / "train_annot",
    "val_annot": OUTPUT_DIR / "val_annot",
    "test_annot": OUTPUT_DIR / "test_annot",
}

for directory in output_dirs.values():
    directory.mkdir(parents=True, exist_ok=True)
    
    
images = sorted(IMAGE_DIR.glob("[!._]*.png"))

pairs = []

for image_path in images:
    annotation_path = ANNOTATION_DIR / f"{image_path.stem}.txt"

    if not annotation_path.exists():
        print(f"WARNING: Missing annotation for {image_path.name}")
        continue

    pairs.append((image_path, annotation_path))


if not pairs:
    raise RuntimeError("No matching image/annotation pairs were found.")

random.seed(RANDOM_SEED)
random.shuffle(pairs)

total = len(pairs)
train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_pairs = pairs[:train_end]
val_pairs = pairs[train_end:val_end]
test_pairs = pairs[val_end:]


def copy_pairs(pairs, image_output, annotation_output):
    for image_path, annotation_path in pairs:
        shutil.copy2(
            image_path,
            image_output / image_path.name
        )

        shutil.copy2(
            annotation_path,
            annotation_output / annotation_path.name
        )

copy_pairs(
    train_pairs,
    output_dirs["train_img"],
    output_dirs["train_annot"]
)

copy_pairs(
    val_pairs,
    output_dirs["val_img"],
    output_dirs["val_annot"]
)

copy_pairs(
    test_pairs,
    output_dirs["test_img"],
    output_dirs["test_annot"]
)


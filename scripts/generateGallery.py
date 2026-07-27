from pathlib import Path
import random,os
from dotenv import load_dotenv
from src.preprocessing.visualize_dota import visualize_single
load_dotenv()

LABEL_DIR=os.getenv('ANOT_TRAIN')
IMG_DIR=os.getenv('TRAIN_DIR')

def main():
    images_dir = Path(IMG_DIR)
    labels_dir = Path(LABEL_DIR)
    out_dir = Path("assets/figures/dota_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(images_dir.glob("[!._]*.png")) + list(images_dir.glob("[!._]*.jpg")))
    random.seed(42)
    sample_files = random.sample(image_files, k=min(24, len(image_files)))

    for idx, image_path in enumerate(sample_files, start=1):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        out_path = out_dir / f"sample_{idx:03d}_{image_path.stem}.png"
        visualize_single(image_path, label_path, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
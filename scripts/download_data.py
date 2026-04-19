"""
Download benchmark datasets for GS-CLIP evaluation.

Usage:
    # Download everything
    python scripts/download_data.py

    # Download individual datasets
    python scripts/download_data.py --dataset winoground   # requires HF_TOKEN
    python scripts/download_data.py --dataset sugarcrepe   # free download

Note on Visual Genome (training data):
    VG (~15GB) must be downloaded manually from https://visualgenome.org/
    - scene_graphs.json
    - VG_100K/ and VG_100K_2/ image directories
    Place them under data/visual_genome/.

Note on MS-COCO (Stage II training data):
    Download from https://cocodataset.org/#download
    - train2017.zip (images)
    - annotations_trainval2017.zip
    Place under data/coco/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gs_clip.data.datasets import WinogroundDataset, SugarCrepeDataset


def download_winoground(data_dir: str) -> None:
    print("\n=== Winoground ===")
    print("Requires HF_TOKEN environment variable.")
    print("Get a token at: https://huggingface.co/settings/tokens")
    print("Accept dataset terms: https://huggingface.co/datasets/facebook/winoground")
    WinogroundDataset(data_dir=data_dir, download=True)


def download_sugarcrepe(data_dir: str) -> None:
    print("\n=== SugarCrepe ===")
    # Downloading the first split triggers download of all splits
    SugarCrepeDataset(data_dir=data_dir, split="replace_rel", download=True)


def print_manual_instructions() -> None:
    print("\n=== Manual downloads ===")
    print("""
Visual Genome (Stage I training, ~15 GB):
  1. Visit https://visualgenome.org/api/v0/api_home.html
  2. Download:
       - scene_graphs.json
       - VG_100K.zip
       - VG_100K_2.zip
  3. Extract to data/visual_genome/

MS-COCO (Stage II training, ~19 GB):
  1. Visit https://cocodataset.org/#download
  2. Download:
       - 2017 Train images [118K/18GB]
       - 2017 Val images [5K/1GB]
       - 2017 Train/Val annotations
  3. Extract to data/coco/
""")


def parse_args():
    p = argparse.ArgumentParser(description="Download GS-CLIP benchmark datasets")
    p.add_argument("--data_dir",  default="data")
    p.add_argument("--dataset",   choices=["winoground", "sugarcrepe", "all"],
                                  default="all")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_dir)

    if args.dataset in ("sugarcrepe", "all"):
        try:
            download_sugarcrepe(str(root / "sugarcrepe"))
        except Exception as e:
            print(f"SugarCrepe failed: {e}")

    if args.dataset in ("winoground", "all"):
        try:
            download_winoground(str(root / "winoground"))
        except Exception as e:
            print(f"Winoground failed: {e}")

    print_manual_instructions()
    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build an isolated supervised LVIS key dataset with hard negatives.

The script keeps source images separated by split, crops around every annotated
key with useful surrounding context, and writes every key contained in a crop.
This is intentionally based on annotations rather than image-search guesses.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ijson
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAILY_DATASET_ROOT = PROJECT_ROOT / "data/openimages_daily_objects"
DATASET_ROOT = PROJECT_ROOT / "data/lvis_keys"
# Reuse the already-downloaded annotation cache from the daily-object build.
METADATA_CACHE_ROOT = DAILY_DATASET_ROOT / "metadata"
SOURCE_ROOT = DAILY_DATASET_ROOT / "lvis_key_sources"
KEY_CLASS_ID = 0
KEY_CATEGORY_ID = 605
SEED = 42

ANNOTATION_URLS = {
    "train": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
    "val": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "VisionGuard-dataset-builder/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def _ensure_annotations(split: str) -> Path:
    destination = METADATA_CACHE_ROOT / f"lvis_v1_{split}.json.zip"
    if not destination.is_file():
        print(f"downloading {ANNOTATION_URLS[split]}")
        _download(ANNOTATION_URLS[split], destination)
    return destination


def _member_name(split: str) -> str:
    return f"lvis_v1_{split}.json"


def _read_key_annotations(archive: Path, split: str) -> dict[int, list[dict]]:
    annotations: dict[int, list[dict]] = defaultdict(list)
    with zipfile.ZipFile(archive) as zipped, zipped.open(_member_name(split)) as stream:
        for annotation in ijson.items(stream, "annotations.item"):
            if annotation["category_id"] != KEY_CATEGORY_ID:
                continue
            annotations[int(annotation["image_id"])].append(
                {
                    "id": int(annotation["id"]),
                    "bbox": [float(value) for value in annotation["bbox"]],
                }
            )
    return dict(annotations)


def _read_images(archive: Path, split: str, image_ids: set[int]) -> dict[int, dict]:
    images: dict[int, dict] = {}
    with zipfile.ZipFile(archive) as zipped, zipped.open(_member_name(split)) as stream:
        for image in ijson.items(stream, "images.item"):
            image_id = int(image["id"])
            if image_id not in image_ids:
                continue
            images[image_id] = {
                "id": image_id,
                "width": int(image["width"]),
                "height": int(image["height"]),
                # The canonical COCO hostname currently serves a mismatched TLS
                # certificate. Its official S3 bucket contains the same object
                # keys with a valid certificate and the exact annotated pixels.
                "url": str(image["coco_url"]).replace(
                    "http://images.cocodataset.org/",
                    "https://s3.amazonaws.com/images.cocodataset.org/",
                    1,
                ),
            }
            if len(images) == len(image_ids):
                break
    missing = image_ids - images.keys()
    if missing:
        raise RuntimeError(f"LVIS image metadata is missing {len(missing)} key images")
    return images


def _download_sources(images: dict[int, dict], split: str, workers: int) -> dict[int, Path]:
    split_root = SOURCE_ROOT / split
    split_root.mkdir(parents=True, exist_ok=True)

    def fetch(image: dict) -> tuple[int, Path]:
        destination = split_root / f"{image['id']:012d}.jpg"
        if not destination.is_file() or destination.stat().st_size == 0:
            _download(image["url"], destination)
        return image["id"], destination

    paths: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, image) for image in images.values()]
        for index, future in enumerate(as_completed(futures), start=1):
            image_id, path = future.result()
            paths[image_id] = path
            if index % 25 == 0 or index == len(futures):
                print(f"{split}: downloaded {index}/{len(futures)} source images")
    return paths


def _square_crop(bbox: list[float], width: int, height: int, scale: float) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = bbox
    center_x = x + box_width / 2
    center_y = y + box_height / 2
    side = min(max(max(box_width, box_height) * scale, 160.0), float(max(width, height)))
    left = max(0.0, min(center_x - side / 2, width - side))
    top = max(0.0, min(center_y - side / 2, height - side))
    right = min(float(width), left + side)
    bottom = min(float(height), top + side)
    return round(left), round(top), round(right), round(bottom)


def _intersection_ratio(bbox: list[float], crop: tuple[int, int, int, int]) -> float:
    x, y, width, height = bbox
    left, top, right, bottom = crop
    intersection = max(0.0, min(x + width, right) - max(x, left)) * max(
        0.0, min(y + height, bottom) - max(y, top)
    )
    return intersection / max(width * height, 1.0)


def _yolo_line(bbox: list[float], crop: tuple[int, int, int, int]) -> str:
    x, y, width, height = bbox
    left, top, right, bottom = crop
    clipped_left = max(x, left)
    clipped_top = max(y, top)
    clipped_right = min(x + width, right)
    clipped_bottom = min(y + height, bottom)
    crop_width = right - left
    crop_height = bottom - top
    center_x = ((clipped_left + clipped_right) / 2 - left) / crop_width
    center_y = ((clipped_top + clipped_bottom) / 2 - top) / crop_height
    normalized_width = (clipped_right - clipped_left) / crop_width
    normalized_height = (clipped_bottom - clipped_top) / crop_height
    return (
        f"{KEY_CLASS_ID} {center_x:.6f} {center_y:.6f} "
        f"{normalized_width:.6f} {normalized_height:.6f}"
    )


def _write_crops(
    split: str,
    images: dict[int, dict],
    annotations: dict[int, list[dict]],
    sources: dict[int, Path],
    scale: float,
    max_crops_per_image: int,
) -> int:
    image_root = DATASET_ROOT / "images" / split
    label_root = DATASET_ROOT / "labels" / split
    image_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_id in sorted(images):
        with Image.open(sources[image_id]) as source:
            source = source.convert("RGB")
            targets = sorted(
                annotations[image_id],
                key=lambda item: item["bbox"][2] * item["bbox"][3],
                reverse=True,
            )[:max_crops_per_image]
            for target in targets:
                crop_box = _square_crop(target["bbox"], source.width, source.height, scale)
                labels = [
                    _yolo_line(candidate["bbox"], crop_box)
                    for candidate in annotations[image_id]
                    if _intersection_ratio(candidate["bbox"], crop_box) >= 0.90
                ]
                if not labels:
                    continue
                stem = f"lvis_key_{image_id:012d}_{target['id']:012d}"
                source.crop(crop_box).save(image_root / f"{stem}.jpg", quality=94)
                (label_root / f"{stem}.txt").write_text("\n".join(labels) + "\n")
                count += 1
    return count


def _update_dataset_yaml() -> None:
    yaml_path = DATASET_ROOT / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        f"path: {DATASET_ROOT}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: key\n"
    )


def _remove_generated_crops() -> None:
    """Remove only crops generated by this script before rebuilding splits."""
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            for prefix in ("lvis_key_*", "negative_*"):
                for path in (DATASET_ROOT / kind / split).glob(prefix):
                    path.unlink()


def _add_hard_negatives(split: str, count: int) -> int:
    """Add deterministic pen/bottle images with empty labels as hard negatives."""
    candidates: list[Path] = []
    label_root = DAILY_DATASET_ROOT / "labels" / split
    for label_path in sorted(label_root.glob("*.txt")):
        class_ids = {line.split(maxsplit=1)[0] for line in label_path.read_text().splitlines()}
        if class_ids & {"2", "8"}:  # bottle and pen in the daily-object dataset
            image_path = DAILY_DATASET_ROOT / "images" / split / f"{label_path.stem}.jpg"
            if image_path.is_file():
                candidates.append(image_path)
    random.Random(SEED + {"train": 0, "val": 1, "test": 2}[split]).shuffle(candidates)
    image_root = DATASET_ROOT / "images" / split
    label_output = DATASET_ROOT / "labels" / split
    for source in candidates[:count]:
        destination = image_root / f"negative_{source.name}"
        shutil.copy2(source, destination)
        (label_output / f"negative_{source.stem}.txt").write_text("")
    return min(count, len(candidates))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--crop-scale", type=float, default=6.0)
    parser.add_argument("--train-images", type=int, default=120)
    parser.add_argument("--val-images", type=int, default=25)
    parser.add_argument("--max-crops-per-image", type=int, default=1)
    parser.add_argument("--train-negatives", type=int, default=80)
    parser.add_argument("--eval-negatives", type=int, default=20)
    args = parser.parse_args()
    if (
        args.workers < 1
        or args.crop_scale < 2
        or args.train_images < 1
        or args.val_images < 1
        or args.max_crops_per_image < 1
        or args.train_negatives < 0
        or args.eval_negatives < 0
    ):
        parser.error("counts must be positive and crop-scale must be at least 2")

    _remove_generated_crops()
    summary: dict[str, dict] = {}
    for source_split in ("train", "val"):
        archive = _ensure_annotations(source_split)
        annotations = _read_key_annotations(archive, source_split)
        images = _read_images(archive, source_split, set(annotations))
        sources = _download_sources(images, source_split, args.workers)

        if source_split == "train":
            image_ids = sorted(images)
            random.Random(SEED).shuffle(image_ids)
            assignments = {"train": sorted(image_ids[: args.train_images])}
        else:
            image_ids = sorted(images)
            random.Random(SEED).shuffle(image_ids)
            assignments = {
                "val": sorted(image_ids[: args.val_images]),
                "test": sorted(image_ids[args.val_images :]),
            }

        for target_split, image_ids in assignments.items():
            selected_images = {image_id: images[image_id] for image_id in image_ids}
            selected_annotations = {image_id: annotations[image_id] for image_id in image_ids}
            crop_count = _write_crops(
                target_split,
                selected_images,
                selected_annotations,
                sources,
                args.crop_scale,
                args.max_crops_per_image,
            )
            summary[target_split] = {
                "source_images": len(image_ids),
                "crops": crop_count,
                "instances": sum(len(selected_annotations[image_id]) for image_id in image_ids),
            }

    _update_dataset_yaml()
    for split in ("train", "val", "test"):
        negative_count = args.train_negatives if split == "train" else args.eval_negatives
        summary[split]["hard_negatives"] = _add_hard_negatives(split, negative_count)
    provenance = {
        "dataset": "LVIS v1 (COCO 2017 images)",
        "category": {"id": KEY_CATEGORY_ID, "name": "key"},
        "seed": SEED,
        "crop_scale": args.crop_scale,
        "max_crops_per_image": args.max_crops_per_image,
        "splits": summary,
        "annotation_urls": ANNOTATION_URLS,
    }
    metadata_root = DATASET_ROOT / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    (metadata_root / "lvis_key_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

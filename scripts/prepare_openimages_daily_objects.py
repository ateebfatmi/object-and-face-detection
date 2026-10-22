#!/usr/bin/env python3
"""Download a balanced, annotated daily-object subset of Open Images V7."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "openimages_daily_objects"
CLASS_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
ANNOTATION_URLS = {
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/{source}/{image_id}.jpg"

# Stable class order becomes the YOLO class-ID mapping.
REQUESTED_CLASSES = (
    "Backpack",
    "Baseball bat",
    "Bottle",
    "Ceiling fan",
    "Coffee cup",
    "Headphones",
    "Laptop",
    "Mobile phone",
    "Pen",
    "Watch",
)
CLASS_SOURCES = {
    name: (("Ceiling fan", "Mechanical fan") if name == "Ceiling fan" else (name,))
    for name in REQUESTED_CLASSES
}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "VisionGuard-dataset-builder/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def class_ids(class_csv: Path) -> tuple[dict[str, int], dict[str, str]]:
    names_to_mid: dict[str, str] = {}
    with class_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            names_to_mid[row["DisplayName"]] = row["LabelName"]
    source_names = [source for names in CLASS_SOURCES.values() for source in names]
    missing = [name for name in source_names if name not in names_to_mid]
    if missing:
        raise RuntimeError(f"Open Images is missing requested classes: {', '.join(missing)}")
    mid_to_index = {
        names_to_mid[source]: index
        for index, name in enumerate(REQUESTED_CLASSES)
        for source in CLASS_SOURCES[name]
    }
    mid_to_name = {
        names_to_mid[source]: name
        for name in REQUESTED_CLASSES
        for source in CLASS_SOURCES[name]
    }
    return mid_to_index, mid_to_name


def read_annotations(
    files: dict[str, Path], mid_to_index: dict[str, int]
) -> tuple[dict[tuple[str, str], list[tuple[int, float, float, float, float]]], dict[str, dict[int, list[str]]]]:
    boxes: dict[tuple[str, str], list[tuple[int, float, float, float, float]]] = defaultdict(list)
    candidates: dict[str, dict[int, list[str]]] = {
        source: defaultdict(list) for source in files
    }
    for source, path in files.items():
        seen: dict[int, set[str]] = defaultdict(set)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                class_index = mid_to_index.get(row["LabelName"])
                if class_index is None:
                    continue
                if row["IsGroupOf"] == "1" or row["IsDepiction"] == "1" or row["IsInside"] == "1":
                    continue
                image_id = row["ImageID"]
                xmin, xmax = float(row["XMin"]), float(row["XMax"])
                ymin, ymax = float(row["YMin"]), float(row["YMax"])
                if not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
                    continue
                boxes[(source, image_id)].append((class_index, xmin, xmax, ymin, ymax))
                seen[class_index].add(image_id)
        for class_index, image_ids in seen.items():
            candidates[source][class_index] = sorted(image_ids)
    return boxes, candidates


def choose_images(
    candidates: dict[str, dict[int, list[str]]], per_class: int, seed: int
) -> dict[tuple[str, str], str]:
    rng = random.Random(seed)
    assignments: dict[tuple[str, str], str] = {}
    def fill(pool: list[tuple[str, str]], class_index: int, split: str, target: int) -> None:
        if target <= 0:
            return
        accepted = 0
        for key in pool:
            assigned = assignments.get(key)
            if assigned is not None and assigned != split:
                continue
            assignments[key] = split
            accepted += 1
            if accepted >= target:
                return
        print(f"warning: class {REQUESTED_CLASSES[class_index]!r} has only {accepted}/{target} images for {split}")

    for class_index in range(len(REQUESTED_CLASSES)):
        pool = [
            (source, image_id)
            for source in sorted(candidates)
            for image_id in candidates[source].get(class_index, ())
        ]
        rng.shuffle(pool)
        total = min(per_class, len(pool))
        if total < 3:
            print(f"warning: class {REQUESTED_CLASSES[class_index]!r} has only {total} total images")
        val_target = 1 if total >= 3 else 0
        test_target = 1 if total >= 2 else 0
        if total >= 10:
            val_target = max(1, round(total * 0.15))
            test_target = max(1, round(total * 0.15))
        train_target = max(0, total - val_target - test_target)
        # Reserve held-out examples before the larger training allocation.
        fill(pool, class_index, "val", val_target)
        fill(pool, class_index, "test", test_target)
        fill(pool, class_index, "train", train_target)
    return assignments


def write_dataset(
    root: Path,
    assignments: dict[tuple[str, str], str],
    boxes: dict[tuple[str, str], list[tuple[int, float, float, float, float]]],
    workers: int,
) -> None:
    cache = root / "image_cache"
    cache.mkdir(parents=True, exist_ok=True)
    # Reuse images from interrupted or differently split runs without leaving
    # stale unlabeled files in a YOLO image directory.
    for existing in root.glob("images/*/*.jpg"):
        cached = cache / existing.name
        if cached.exists():
            existing.unlink()
        else:
            existing.replace(cached)
    for partial in root.glob("images/*/*.part"):
        partial.unlink()
    shutil.rmtree(root / "labels", ignore_errors=True)
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    def fetch(item: tuple[tuple[str, str], str]) -> tuple[tuple[str, str], str, str | None]:
        (source, image_id), split = item
        destination = root / "images" / split / f"{image_id}.jpg"
        cached = cache / f"{image_id}.jpg"
        try:
            download(IMAGE_URL.format(source=source, image_id=image_id), cached)
            destination.unlink(missing_ok=True)
            try:
                destination.hardlink_to(cached)
            except OSError:
                shutil.copy2(cached, destination)
            return (source, image_id), split, None
        except Exception as exc:  # network errors are summarized after all workers finish
            destination.unlink(missing_ok=True)
            return (source, image_id), split, str(exc)

    failures: list[tuple[str, str]] = []
    completed: list[tuple[tuple[str, str], str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, item): item for item in assignments.items()}
        for index, future in enumerate(as_completed(futures), start=1):
            key, split, error = future.result()
            if error:
                failures.append((f"{key[0]}/{key[1]}", error))
            else:
                completed.append((key, split))
            if index % 50 == 0 or index == len(futures):
                print(f"downloaded {index - len(failures)}/{len(futures)} images")

    if failures:
        sample = "; ".join(f"{key}: {error}" for key, error in failures[:3])
        raise RuntimeError(f"{len(failures)} image downloads failed. Examples: {sample}")

    manifest = []
    class_counts: Counter[str] = Counter()
    for (source, image_id), split in sorted(completed):
        label_path = root / "labels" / split / f"{image_id}.txt"
        lines = []
        # Open Images can contain an exact duplicate annotation row.  Keep the
        # generated labels deterministic and avoid making the trainer silently
        # discard duplicates while building its cache.
        for class_index, xmin, xmax, ymin, ymax in sorted(set(boxes[(source, image_id)])):
            center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
            width, height = xmax - xmin, ymax - ymin
            lines.append(f"{class_index} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
            class_counts[REQUESTED_CLASSES[class_index]] += 1
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        manifest.append({
            "image_id": image_id,
            "openimages_source_split": source,
            "local_split": split,
            "source": IMAGE_URL.format(source=source, image_id=image_id),
        })

    yaml_lines = [
        f"path: {root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *[f"  {index}: {name.lower()}" for index, name in enumerate(REQUESTED_CLASSES)],
        "",
    ]
    (root / "dataset.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")
    (root / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps({
        "images": len(completed),
        "boxes_by_class": dict(class_counts),
        "classes": list(REQUESTED_CLASSES),
    }, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--images-per-class", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.images_per_class < 10:
        parser.error("--images-per-class must be at least 10")
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32")

    metadata = args.output / "metadata"
    class_file = metadata / "class-descriptions.csv"
    download(CLASS_URL, class_file)
    annotation_files = {
        source: metadata / f"{source}-annotations-bbox.csv"
        for source in ANNOTATION_URLS
    }
    for source, url in ANNOTATION_URLS.items():
        print(f"preparing {source} annotations")
        download(url, annotation_files[source])

    mid_to_index, _ = class_ids(class_file)
    boxes, candidates = read_annotations(annotation_files, mid_to_index)
    assignments = choose_images(candidates, args.images_per_class, args.seed)
    print(f"selected {len(assignments)} unique images")
    write_dataset(args.output, assignments, boxes, args.workers)
    print(f"dataset ready: {args.output / 'dataset.yaml'}")


if __name__ == "__main__":
    main()

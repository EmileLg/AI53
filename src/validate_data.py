from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

EXPECTED_SPLITS = {"train": 1034, "validation": 133, "test": 128}
EXPECTED_CLASSES = {"angular_leaf_spot", "bean_rust", "healthy"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_dataset(raw_root: Path) -> dict:
    """Validate the raw dataset and return a deterministic quality report.

    Required Core checks:
      1. required splits exist;
      2. exactly the expected classes exist in every split;
      3. expected image counts are present;
      4. every image can be fully decoded by Pillow (`Image.load()` or equivalent);
      5. no split or class is empty.

    The report must not contain timestamps because it is a DVC output.
    Any failed check raises an error so the DVC pipeline stops.
    """
    raw_root = Path(raw_root)
    expected_classes = sorted(EXPECTED_CLASSES)
    splits_report = {}

    for split, expected_count in EXPECTED_SPLITS.items():
        # 1. The split folder must exist.
        split_dir = raw_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split folder: {split_dir}")

        # 2. Exactly the expected classes, no missing or extra folders.
        found_classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        if found_classes != expected_classes:
            raise RuntimeError(
                f"Split '{split}' has classes {found_classes}, expected {expected_classes}"
            )

        class_counts = {}
        for class_name in expected_classes:
            images = sorted(
                p for p in (split_dir / class_name).rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            # 5. No empty class.
            if not images:
                raise RuntimeError(f"Class '{class_name}' in split '{split}' is empty")

            # 4. Decode every pixel, not just the header. Errors (OSError,
            # UnidentifiedImageError) are passed on so the pipeline stops.
            for image_path in images:
                with Image.open(image_path) as img:
                    img.load()

            class_counts[class_name] = len(images)

        # 3. Image count must match the course archives.
        split_total = sum(class_counts.values())
        if split_total != expected_count:
            raise RuntimeError(
                f"Split '{split}' has {split_total} images, expected {expected_count}"
            )

        splits_report[split] = {"images": split_total, "class_counts": class_counts}

    # 6. Deterministic: sorted lists, counts only, no timestamps or paths.
    return {
        "status": "PASS",
        "classes": expected_classes,
        "splits": splits_report,
        "total_images": sum(s["images"] for s in splits_report.values()),
    }


def main() -> None:
    report = validate_dataset(Path("data/raw"))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

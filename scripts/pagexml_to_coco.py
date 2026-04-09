#!/usr/bin/env python3
"""Convert PAGE XML + images into per-image COCO JSON files for DocSAM.

This script recursively scans a dataset root and finds all PAGE XML files located in
folders named "page". For each PAGE XML file, it looks up an image with the same stem
anywhere under the same dataset root, then writes a COCO annotation JSON in the
output folder.

The generated JSON structure matches the per-image layout expected by DocSAM:

    <dataset_root>/coco/<stem>.json

Each JSON contains one image entry and all region annotations from the PAGE XML.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET
import imghdr
import struct


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}


@dataclass
class RegionAnnotation:
    label: str
    segmentation: List[float]
    bbox: List[float]
    area: float


@dataclass
class ParsedPage:
    width: Optional[int]
    height: Optional[int]
    annotations: List[RegionAnnotation]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PAGE XML files and matching images into DocSAM COCO JSON annotations."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Root directory containing PAGE XML files in folders named 'page' and images anywhere below this root.",
    )
    parser.add_argument(
        "--output-coco-dir",
        type=Path,
        default=None,
        help="Output directory for generated JSON files (default: <dataset_root>/coco).",
    )
    parser.add_argument(
        "--output-list",
        type=Path,
        default=None,
        help="Optional path to write list.txt (one image filename per line).",
    )
    parser.add_argument(
        "--image-dir-name",
        type=str,
        default="image",
        help="Relative image directory name to write inside each COCO 'file_name' (default: image).",
    )
    parser.add_argument(
        "--background-name",
        type=str,
        default="_background_",
        help="Category name to reserve as id=0 (default: _background_).",
    )
    parser.add_argument(
        "--default-label",
        type=str,
        default="text",
        help="Fallback class label when PAGE region type cannot be found (default: text).",
    )
    parser.add_argument(
        "--allow-duplicate-stems",
        action="store_true",
        help="Allow duplicate image stems and match the nearest path to each PAGE XML.",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help=(
            "Run a dry validation pass on XML/image pairs and generated COCO payloads "
            "without writing JSON files."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of PAGE XML files to process (useful for quick checks).",
    )
    return parser.parse_args()


def strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def parse_points(points_str: str) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for token in points_str.strip().split():
        if "," not in token:
            continue
        x_str, y_str = token.split(",", 1)
        try:
            points.append((float(x_str), float(y_str)))
        except ValueError:
            continue
    return points


def polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    accum = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        accum += x1 * y2 - x2 * y1
    return abs(accum) * 0.5


def sanitize_label(raw: str, default_label: str) -> str:
    raw = (raw or "").strip().lower()
    if not raw:
        return default_label
    cleaned = re.sub(r"\s+", " ", raw)
    return cleaned


def infer_region_label(region_el: ET.Element, default_label: str) -> str:
    for attr_name in ("type", "custom", "regionType"):
        if region_el.get(attr_name):
            value = region_el.get(attr_name, "")
            if attr_name == "custom":
                match = re.search(r"type\s*:\s*([^;}]*)", value)
                if match:
                    return sanitize_label(match.group(1), default_label)
            return sanitize_label(value, default_label)
    return default_label


def find_child_by_local_name(parent: ET.Element, local_name: str) -> Optional[ET.Element]:
    for child in list(parent):
        if strip_namespace(child.tag) == local_name:
            return child
    return None


def region_elements(page_el: ET.Element) -> Iterable[ET.Element]:
    for el in page_el.iter():
        tag = strip_namespace(el.tag)
        if tag.endswith("Region") and tag != "Page":
            yield el


def parse_pagexml(xml_path: Path, default_label: str) -> ParsedPage:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    page_el: Optional[ET.Element] = None
    for el in root.iter():
        if strip_namespace(el.tag) == "Page":
            page_el = el
            break
    if page_el is None:
        raise ValueError(f"No <Page> element found in {xml_path}")

    page_width = page_el.get("imageWidth")
    page_height = page_el.get("imageHeight")
    width = int(page_width) if page_width and page_width.isdigit() else None
    height = int(page_height) if page_height and page_height.isdigit() else None

    annotations: List[RegionAnnotation] = []
    for region_el in region_elements(page_el):
        coords_el = find_child_by_local_name(region_el, "Coords")
        if coords_el is None:
            continue
        points_str = coords_el.get("points", "")
        points = parse_points(points_str)
        if len(points) < 3:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        segmentation: List[float] = [coord for point in points for coord in point]
        area = polygon_area(points)
        if area <= 0:
            continue

        label = infer_region_label(region_el, default_label)
        annotations.append(
            RegionAnnotation(
                label=label,
                segmentation=segmentation,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=area,
            )
        )

    return ParsedPage(width=width, height=height, annotations=annotations)


def image_size_from_header(image_path: Path) -> Tuple[int, int]:
    image_type = imghdr.what(image_path)
    with image_path.open("rb") as f:
        data = f.read(32)

    if image_type == "png":
        if len(data) < 24:
            raise ValueError(f"PNG header too short: {image_path}")
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    if image_type == "gif":
        if len(data) < 10:
            raise ValueError(f"GIF header too short: {image_path}")
        width, height = struct.unpack("<HH", data[6:10])
        return int(width), int(height)
    if image_type == "jpeg":
        with image_path.open("rb") as f:
            f.read(2)
            while True:
                marker_prefix = f.read(1)
                if marker_prefix != b"\xFF":
                    continue
                marker_type = f.read(1)
                while marker_type == b"\xFF":
                    marker_type = f.read(1)
                if marker_type in {b"\xD8", b"\xD9"}:
                    continue
                seg_len_bytes = f.read(2)
                if len(seg_len_bytes) != 2:
                    break
                seg_len = struct.unpack(">H", seg_len_bytes)[0]
                if marker_type in {
                    b"\xC0",
                    b"\xC1",
                    b"\xC2",
                    b"\xC3",
                    b"\xC5",
                    b"\xC6",
                    b"\xC7",
                    b"\xC9",
                    b"\xCA",
                    b"\xCB",
                    b"\xCD",
                    b"\xCE",
                    b"\xCF",
                }:
                    f.read(1)
                    height, width = struct.unpack(">HH", f.read(4))
                    return int(width), int(height)
                f.seek(seg_len - 2, 1)
        raise ValueError(f"Unable to parse JPEG size: {image_path}")

    raise ValueError(
        f"Unsupported image format for header parsing ({image_type}): {image_path}"
    )


def index_images(root: Path) -> Dict[str, List[Path]]:
    image_map: Dict[str, List[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        image_map.setdefault(path.stem, []).append(path)
    return image_map


def find_pagexml_files(root: Path) -> List[Path]:
    return sorted(
        p
        for p in root.rglob("*.xml")
        if p.is_file() and p.parent.name.lower() == "page"
    )


def choose_best_image_match(xml_path: Path, matches: List[Path]) -> Path:
    # Prefer image path with the longest common prefix (closest in directory tree).
    xml_parts = xml_path.resolve().parts

    def score(img_path: Path) -> int:
        img_parts = img_path.resolve().parts
        common = 0
        for a, b in zip(xml_parts, img_parts):
            if a != b:
                break
            common += 1
        return common

    return sorted(matches, key=score, reverse=True)[0]


def build_coco_for_image(
    image_path: Path,
    xml_width: Optional[int],
    xml_height: Optional[int],
    image_dir_name: str,
    annotations: Sequence[RegionAnnotation],
    category_to_id: Dict[str, int],
) -> dict:
    if xml_width is not None and xml_height is not None:
        width, height = xml_width, xml_height
    else:
        width, height = image_size_from_header(image_path)

    coco_annotations = []
    for ann_id, ann in enumerate(annotations, start=1):
        coco_annotations.append(
            {
                "id": ann_id,
                "image_id": 1,
                "category_id": category_to_id[ann.label],
                "bbox": [round(v, 3) for v in ann.bbox],
                "segmentation": [[round(v, 3) for v in ann.segmentation]],
                "area": round(ann.area, 3),
                "iscrowd": 0,
            }
        )

    categories = [
        {"id": idx, "name": label, "supercategory": ""}
        for label, idx in sorted(category_to_id.items(), key=lambda x: x[1])
    ]

    return {
        "images": [
            {
                "id": 1,
                "file_name": str(Path(image_dir_name) / image_path.name),
                "width": width,
                "height": height,
            }
        ],
        "annotations": coco_annotations,
        "categories": categories,
    }


def sanity_check_coco(coco: dict, source_xml: Path, image_path: Path) -> None:
    if "images" not in coco or len(coco["images"]) != 1:
        raise ValueError(f"{source_xml}: expected exactly 1 image entry")
    image = coco["images"][0]
    required_image_keys = {"id", "file_name", "width", "height"}
    missing_image_keys = required_image_keys - set(image.keys())
    if missing_image_keys:
        raise ValueError(f"{source_xml}: missing image keys: {sorted(missing_image_keys)}")
    if image["width"] <= 0 or image["height"] <= 0:
        raise ValueError(f"{source_xml}: invalid image size from {image_path}")

    category_ids = set()
    for cat in coco.get("categories", []):
        if "id" not in cat or "name" not in cat:
            raise ValueError(f"{source_xml}: invalid category entry: {cat}")
        category_ids.add(cat["id"])

    for ann in coco.get("annotations", []):
        for key in ("id", "image_id", "category_id", "bbox", "segmentation", "area", "iscrowd"):
            if key not in ann:
                raise ValueError(f"{source_xml}: annotation missing key '{key}'")
        if ann["category_id"] not in category_ids:
            raise ValueError(f"{source_xml}: annotation category_id {ann['category_id']} not in categories")
        if len(ann["bbox"]) != 4:
            raise ValueError(f"{source_xml}: annotation bbox should have 4 values")
        if not ann["segmentation"] or not isinstance(ann["segmentation"], list):
            raise ValueError(f"{source_xml}: annotation segmentation must be a non-empty list")
        if ann["area"] < 0:
            raise ValueError(f"{source_xml}: annotation area must be non-negative")


def main() -> int:
    args = parse_args()

    root = args.dataset_root.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"dataset_root is not a directory: {root}")

    output_coco_dir = (args.output_coco_dir or (root / "coco")).resolve()
    output_coco_dir.mkdir(parents=True, exist_ok=True)

    pagexml_files = find_pagexml_files(root)
    if args.max_files is not None:
        if args.max_files <= 0:
            raise ValueError("--max-files must be > 0")
        pagexml_files = pagexml_files[: args.max_files]
    if not pagexml_files:
        raise FileNotFoundError(
            f"No PAGE XML files found under {root} in subfolders named 'page'."
        )

    image_map = index_images(root)

    if not args.allow_duplicate_stems:
        duplicate_stems = {stem: paths for stem, paths in image_map.items() if len(paths) > 1}
        if duplicate_stems:
            stems = ", ".join(sorted(duplicate_stems.keys())[:10])
            raise ValueError(
                "Duplicate image stems detected. Re-run with --allow-duplicate-stems "
                f"or deduplicate filenames. Example stems: {stems}"
            )

    all_labels = set([args.background_name])
    parsed: List[Tuple[Path, Path, ParsedPage]] = []
    missing_images: List[Path] = []

    for xml_path in pagexml_files:
        image_matches = image_map.get(xml_path.stem, [])
        if not image_matches:
            missing_images.append(xml_path)
            continue

        image_path = image_matches[0]
        if len(image_matches) > 1:
            image_path = choose_best_image_match(xml_path, image_matches)

        parsed_page = parse_pagexml(xml_path, args.default_label)
        parsed.append((xml_path, image_path, parsed_page))
        for ann in parsed_page.annotations:
            all_labels.add(ann.label)

    if missing_images:
        sample = "\n".join(f"  - {p}" for p in missing_images[:10])
        raise FileNotFoundError(
            f"Could not find matching images for {len(missing_images)} PAGE XML files.\n{sample}"
        )

    sorted_labels = [args.background_name] + sorted(label for label in all_labels if label != args.background_name)
    category_to_id = {label: idx for idx, label in enumerate(sorted_labels)}

    image_names: List[str] = []
    for xml_path, image_path, parsed_page in parsed:
        coco = build_coco_for_image(
            image_path=image_path,
            xml_width=parsed_page.width,
            xml_height=parsed_page.height,
            image_dir_name=args.image_dir_name,
            annotations=parsed_page.annotations,
            category_to_id=category_to_id,
        )
        sanity_check_coco(coco, source_xml=xml_path, image_path=image_path)
        if args.sanity_check:
            image_names.append(image_path.name)
            continue
        out_json = output_coco_dir / f"{xml_path.stem}.json"
        out_json.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
        image_names.append(image_path.name)

    if args.output_list is not None and not args.sanity_check:
        args.output_list.parent.mkdir(parents=True, exist_ok=True)
        args.output_list.write_text("\n".join(image_names) + "\n", encoding="utf-8")

    if args.sanity_check:
        print(f"Sanity check passed for {len(parsed)} PAGE XML/image pairs.")
    else:
        print(f"Converted {len(parsed)} PAGE XML files -> {output_coco_dir}")
    print(f"Categories ({len(sorted_labels)}): {sorted_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

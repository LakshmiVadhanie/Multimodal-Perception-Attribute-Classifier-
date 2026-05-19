"""
VLM auto-labeling pipeline entry point.

Scans an input directory for images, runs the VLM labeler on each one,
and saves an annotation JSON that can be fed directly into training.

Usage:
    python scripts/run_auto_labeling.py \
        --image_dir /path/to/unlabeled/images \
        --output_dir /path/to/output \
        --vlm_model blip2 \
        --road_user_type pedestrian

    # Resume an interrupted run:
    python scripts/run_auto_labeling.py \
        --image_dir /path/to/unlabeled/images \
        --output_dir /path/to/output \
        --vlm_model blip2 \
        --checkpoint /path/to/output/checkpoint.json
"""

import argparse
import json
from pathlib import Path

import yaml
from loguru import logger

from src.pipeline.vlm_labeler import VLMLabeler
from src.pipeline.label_validator import LabelValidator
from src.dataset.ontology import ROAD_USER_TYPES


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images(image_dir: str) -> list:
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    paths = []
    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(image_dir.rglob(f"*{ext}"))
        paths.extend(image_dir.rglob(f"*{ext.upper()}"))

    paths = sorted(set(str(p) for p in paths))
    logger.info(f"Found {len(paths)} images in {image_dir}")
    return paths


def print_coverage_report(annotations: list):
    validator = LabelValidator()
    coverage = validator.compute_coverage(annotations)

    print("\nLabel Coverage Report")
    print("-" * 40)
    for attr_name, frac in sorted(coverage.items()):
        bar = "#" * int(frac * 20)
        print(f"  {attr_name:<20} {frac:5.1%}  [{bar:<20}]")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="Directory of unlabeled images")
    parser.add_argument("--output_dir", required=True, help="Where to save annotations")
    parser.add_argument("--vlm_model", default="blip2", choices=["blip2", "llava"])
    parser.add_argument(
        "--road_user_type",
        default="vehicle",
        choices=ROAD_USER_TYPES + ["mixed"],
        help="Road user type. Use 'mixed' to infer from filename.",
    )
    parser.add_argument("--config", default=None, help="Optional pipeline_config.yaml path")
    parser.add_argument("--checkpoint", default=None, help="Resume from this checkpoint file")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--confidence_threshold", type=float, default=0.75)
    args = parser.parse_args()

    # load config if provided
    pipeline_cfg = {}
    if args.config:
        with open(args.config) as f:
            pipeline_cfg = yaml.safe_load(f)

    image_paths = collect_images(args.image_dir)
    if not image_paths:
        logger.error("No images found. Exiting.")
        return

    # determine road user type per image
    if args.road_user_type == "mixed":
        # infer from parent directory name or filename keywords
        road_user_types = []
        for path in image_paths:
            p = Path(path)
            combined = (str(p.parent) + p.stem).lower()
            if "pedestrian" in combined or "person" in combined or "ped_" in combined:
                road_user_types.append("pedestrian")
            elif "cyclist" in combined or "bike" in combined or "bicycle" in combined:
                road_user_types.append("cyclist")
            else:
                road_user_types.append("vehicle")
        logger.info(
            f"Mixed mode: {road_user_types.count('vehicle')} vehicle, "
            f"{road_user_types.count('pedestrian')} pedestrian, "
            f"{road_user_types.count('cyclist')} cyclist"
        )
    else:
        road_user_types = [args.road_user_type] * len(image_paths)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = str(output_dir / "auto_labels.json")
    checkpoint_file = args.checkpoint or str(output_dir / "labeling_checkpoint.json")

    labeler = VLMLabeler(
        vlm_model=args.vlm_model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        confidence_threshold=args.confidence_threshold,
    )

    annotations = labeler.run_pipeline(
        image_paths=image_paths,
        road_user_types=road_user_types,
        output_file=output_file,
        checkpoint_file=checkpoint_file,
        save_every_n=pipeline_cfg.get("labeling", {}).get("save_every_n", 100),
    )

    print_coverage_report(annotations)

    # write a summary
    success = sum(1 for a in annotations if a.get("labeling_method") == "vlm_auto")
    failed = sum(1 for a in annotations if a.get("labeling_method") == "failed")
    summary = {
        "total_images": len(annotations),
        "successful": success,
        "failed": failed,
        "vlm_model": args.vlm_model,
        "output_file": output_file,
    }
    summary_path = output_dir / "labeling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Done. {success}/{len(annotations)} images labeled. "
        f"Annotations saved to {output_file}"
    )


if __name__ == "__main__":
    main()

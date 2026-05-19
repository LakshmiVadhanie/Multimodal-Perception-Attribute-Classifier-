"""
Evaluation script for the trained attribute classifier.

Computes per-attribute accuracy, F1 score, and confusion matrices
on the test set. Logs results to MLflow and saves a report.

Usage:
    python scripts/evaluate.py \
        --checkpoint ./checkpoints/best_model \
        --config configs/model_config.yaml \
        --output_dir ./eval_results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.dataset.augmentations import AugmentationConfig, build_val_transforms
from src.dataset.loader import (
    RoadUserDataset,
    load_bdd100k_annotations,
    load_custom_annotations,
    split_annotations,
)
from src.dataset.ontology import LABEL_MAPS, ONTOLOGY
from src.models.vit_classifier import ViTAttributeClassifier
from src.tracking.mlflow_logger import MLflowLogger


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_evaluation(model, loader, device):
    model.eval()
    all_preds = {name: [] for name in ONTOLOGY}
    all_labels = {name: [] for name in ONTOLOGY}

    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(device)
        labels = batch["labels"]

        logits = model(images)

        for attr_name in ONTOLOGY:
            attr_logits = logits[attr_name]
            attr_labels = labels[attr_name]
            valid_mask = attr_labels >= 0

            if valid_mask.sum() == 0:
                continue

            preds = attr_logits[valid_mask].argmax(dim=-1).cpu().numpy()
            true = attr_labels[valid_mask].cpu().numpy()

            all_preds[attr_name].extend(preds.tolist())
            all_labels[attr_name].extend(true.tolist())

    return all_preds, all_labels


def compute_metrics(all_preds, all_labels):
    results = {}
    for attr_name in ONTOLOGY:
        preds = all_preds[attr_name]
        labels = all_labels[attr_name]

        if len(preds) == 0:
            continue

        label_names = list(LABEL_MAPS[attr_name].values())
        report = classification_report(
            labels, preds, target_names=label_names, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(labels, preds).tolist()

        results[attr_name] = {
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "per_class": {
                name: {
                    "precision": report[name]["precision"],
                    "recall": report[name]["recall"],
                    "f1": report[name]["f1-score"],
                    "support": report[name]["support"],
                }
                for name in label_names
                if name in report
            },
            "confusion_matrix": cm,
            "label_names": label_names,
            "n_samples": len(preds),
        }

    return results


def print_summary(results):
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    accs = []
    f1s = []
    for attr_name, metrics in results.items():
        acc = metrics["accuracy"]
        f1 = metrics["macro_f1"]
        accs.append(acc)
        f1s.append(f1)
        print(f"\n[{attr_name}]")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Macro F1 : {f1:.4f}")
        print(f"  Samples  : {metrics['n_samples']}")
        for cls_name, cls_metrics in metrics["per_class"].items():
            print(
                f"    {cls_name:<20} "
                f"P={cls_metrics['precision']:.3f} "
                f"R={cls_metrics['recall']:.3f} "
                f"F1={cls_metrics['f1']:.3f} "
                f"N={cls_metrics['support']}"
            )

    print("\n" + "=" * 60)
    print(f"OVERALL  Accuracy={np.mean(accs):.4f}  Macro F1={np.mean(f1s):.4f}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default="./eval_results")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    log_cfg = cfg["logging"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Evaluating on {device}")

    dataset_format = train_cfg.get("dataset_format", "bdd100k")
    dataset_path = train_cfg["dataset_path"]

    if dataset_format == "bdd100k":
        annotations = load_bdd100k_annotations(dataset_path)
    else:
        annotations = load_custom_annotations(dataset_path)

    _, val_anns, test_anns = split_annotations(
        annotations,
        val_ratio=train_cfg.get("val_split", 0.15),
        test_ratio=train_cfg.get("test_split", 0.10),
        seed=train_cfg.get("seed", 42),
    )

    eval_anns = test_anns if args.split == "test" else val_anns
    aug_config = AugmentationConfig(image_size=train_cfg.get("image_size", 224))
    dataset = RoadUserDataset(eval_anns, dataset_path, transform=build_val_transforms(aug_config))

    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
    )

    model = ViTAttributeClassifier.load(
        path=args.checkpoint,
        model_name=model_cfg["backbone"],
        device=device,
    )
    model = model.to(device)

    all_preds, all_labels = run_evaluation(model, loader, device)
    results = compute_metrics(all_preds, all_labels)
    print_summary(results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Report saved to {report_path}")

    mlflow_logger = MLflowLogger(
        tracking_uri=log_cfg.get("mlflow_tracking_uri", "http://localhost:5000"),
        experiment_name=log_cfg.get("experiment_name", "vit-attribute-classifier"),
        run_name=f"eval-{args.split}",
    )
    mlflow_logger.start_run()
    flat_metrics = {}
    for attr_name, metrics in results.items():
        flat_metrics[f"test/{attr_name}/accuracy"] = metrics["accuracy"]
        flat_metrics[f"test/{attr_name}/macro_f1"] = metrics["macro_f1"]
    all_accs = [m["accuracy"] for m in results.values()]
    flat_metrics["test/mean_accuracy"] = float(np.mean(all_accs))
    mlflow_logger.log_metrics(flat_metrics)
    mlflow_logger.log_artifact(str(report_path))
    mlflow_logger.end_run()


if __name__ == "__main__":
    main()

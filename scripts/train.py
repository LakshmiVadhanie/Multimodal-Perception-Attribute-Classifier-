"""
Training entry point.

Usage:
    python scripts/train.py --config configs/model_config.yaml
    python scripts/train.py --config configs/model_config.yaml --resume ./checkpoints/last
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from loguru import logger
from tqdm import tqdm

from src.dataset.augmentations import AugmentationConfig, build_train_transforms, build_val_transforms
from src.dataset.loader import (
    RoadUserDataset,
    build_dataloaders,
    load_bdd100k_annotations,
    load_custom_annotations,
    split_annotations,
)
from src.dataset.ontology import ONTOLOGY
from src.models.attribute_head import AttributeLoss
from src.models.vit_classifier import ViTAttributeClassifier, build_optimizer, build_scheduler
from src.tracking.mlflow_logger import MLflowLogger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_accuracy(logits: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> Dict[str, float]:
    accuracies = {}
    for attr_name, attr_logits in logits.items():
        attr_labels = labels[attr_name]
        valid_mask = attr_labels >= 0
        if valid_mask.sum() == 0:
            continue
        preds = attr_logits[valid_mask].argmax(dim=-1)
        correct = (preds == attr_labels[valid_mask]).float().mean().item()
        accuracies[attr_name] = correct
    return accuracies


def train_epoch(model, loader, optimizer, scheduler, loss_fn, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_acc = {name: 0.0 for name in ONTOLOGY}
    n_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        labels = {k: v.to(device) for k, v in batch["labels"].items()}

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            losses = loss_fn(logits, labels)
            loss = losses["total"]

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        accs = compute_accuracy(logits, labels)
        total_loss += loss.item()
        for k, v in accs.items():
            total_acc[k] += v
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc = {k: v / max(n_batches, 1) for k, v in total_acc.items()}
    mean_acc = np.mean(list(avg_acc.values()))
    return avg_loss, avg_acc, mean_acc


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_acc = {name: 0.0 for name in ONTOLOGY}
    n_batches = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch["image"].to(device)
        labels = {k: v.to(device) for k, v in batch["labels"].items()}

        logits = model(images)
        losses = loss_fn(logits, labels)
        loss = losses["total"]

        accs = compute_accuracy(logits, labels)
        total_loss += loss.item()
        for k, v in accs.items():
            total_acc[k] += v
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_acc = {k: v / max(n_batches, 1) for k, v in total_acc.items()}
    mean_acc = np.mean(list(avg_acc.values()))
    return avg_loss, avg_acc, mean_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to model_config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    log_cfg = cfg["logging"]

    set_seed(train_cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # load annotations
    dataset_format = train_cfg.get("dataset_format", "bdd100k")
    dataset_path = train_cfg["dataset_path"]

    if dataset_format == "bdd100k":
        annotations = load_bdd100k_annotations(dataset_path)
    else:
        annotations = load_custom_annotations(dataset_path)

    train_anns, val_anns, test_anns = split_annotations(
        annotations,
        val_ratio=train_cfg.get("val_split", 0.15),
        test_ratio=train_cfg.get("test_split", 0.10),
        seed=train_cfg.get("seed", 42),
    )

    aug_cfg_dict = cfg.get("augmentation", {})
    aug_config = AugmentationConfig(
        image_size=train_cfg.get("image_size", 224),
        random_horizontal_flip=aug_cfg_dict.get("random_horizontal_flip", 0.5),
    )

    train_ds = RoadUserDataset(train_anns, dataset_path, transform=build_train_transforms(aug_config))
    val_ds = RoadUserDataset(val_anns, dataset_path, transform=build_val_transforms(aug_config))
    test_ds = RoadUserDataset(test_anns, dataset_path, transform=build_val_transforms(aug_config))

    train_loader, val_loader, test_loader = build_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=train_cfg.get("batch_size", 32),
        num_workers=train_cfg.get("num_workers", 4),
        use_weighted_sampler=train_cfg.get("use_class_weights", False),
    )

    model = ViTAttributeClassifier(
        model_name=model_cfg["backbone"],
        dropout=model_cfg.get("dropout", 0.1),
        freeze_backbone_layers=model_cfg.get("freeze_backbone_layers", 0),
    ).to(device)

    if args.resume:
        model = ViTAttributeClassifier.load(args.resume, model_name=model_cfg["backbone"], device=device)
        model = model.to(device)

    optimizer = build_optimizer(model, lr=train_cfg["learning_rate"], weight_decay=train_cfg.get("weight_decay", 0.01))

    num_training_steps = len(train_loader) * train_cfg["num_epochs"]
    scheduler = build_scheduler(optimizer, train_cfg.get("warmup_steps", 500), num_training_steps)

    loss_fn = AttributeLoss()

    use_amp = train_cfg.get("mixed_precision", True) and device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mlflow_logger = MLflowLogger(
        tracking_uri=log_cfg.get("mlflow_tracking_uri", "http://localhost:5000"),
        experiment_name=log_cfg.get("experiment_name", "vit-attribute-classifier"),
        run_name=log_cfg.get("run_name"),
    )
    mlflow_logger.start_run(params={
        "model": model_cfg["backbone"],
        "batch_size": train_cfg["batch_size"],
        "lr": train_cfg["learning_rate"],
        "epochs": train_cfg["num_epochs"],
        "dataset_format": dataset_format,
        "freeze_layers": model_cfg.get("freeze_backbone_layers", 0),
    })

    best_val_acc = 0.0
    for epoch in range(1, train_cfg["num_epochs"] + 1):
        logger.info(f"Epoch {epoch}/{train_cfg['num_epochs']}")

        train_loss, train_accs, train_mean_acc = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device, scaler
        )
        val_loss, val_accs, val_mean_acc = eval_epoch(model, val_loader, loss_fn, device)

        logger.info(
            f"  train loss={train_loss:.4f} acc={train_mean_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_mean_acc:.4f}"
        )

        metrics = {
            "train/loss": train_loss,
            "train/mean_acc": train_mean_acc,
            "val/loss": val_loss,
            "val/mean_acc": val_mean_acc,
        }
        for attr, acc in val_accs.items():
            metrics[f"val/acc_{attr}"] = acc
        mlflow_logger.log_metrics(metrics, step=epoch)

        # save checkpoint
        ckpt_path = checkpoint_dir / f"epoch_{epoch:02d}"
        model.save(str(ckpt_path))

        if val_mean_acc > best_val_acc:
            best_val_acc = val_mean_acc
            best_path = checkpoint_dir / "best_model"
            model.save(str(best_path))
            logger.info(f"  New best model saved (val_acc={best_val_acc:.4f})")

    mlflow_logger.log_artifact(str(checkpoint_dir / "best_model"), artifact_path="best_model")
    mlflow_logger.end_run()
    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()

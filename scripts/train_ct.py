from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from m2m_alk.ct_model import CTResNet18, load_med3d_pretrained_weights
from m2m_alk.data import NiftiROIDataset, build_ct_transform
from m2m_alk.utils import (
    binary_classification_metrics,
    ensure_dir,
    get_device,
    save_json,
    set_seed,
    write_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CT 3D ResNet18 stream.")
    parser.add_argument("--roi-root", required=True, help="Root containing training/val/external ROI folders.")
    parser.add_argument("--train-metadata", required=True, help="Training CSV/XLSX; default columns: sample_id,label.")
    parser.add_argument("--val-metadata", required=True, help="Internal validation CSV/XLSX.")
    parser.add_argument("--external-metadata", default=None, help="Optional external testing CSV/XLSX.")
    parser.add_argument("--train-split-name", default="training")
    parser.add_argument("--val-split-name", default="val")
    parser.add_argument("--external-split-name", default="external")
    parser.add_argument("--pretrained-weights", default=None, help="Optional Med3D/ResNet18 checkpoint.")
    parser.add_argument("--output-dir", default="outputs/ct_resnet18")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def predict(model, loader, device, threshold: float) -> tuple[dict, pd.DataFrame]:
    model.eval()
    names, probs, labels = [], [], []
    with torch.no_grad():
        for images, y, sample_ids in loader:
            logits = model(images.to(device))
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            names.extend(sample_ids)
            probs.extend(p)
            labels.extend(y.numpy())
    metrics = binary_classification_metrics(labels, probs, threshold=threshold)
    df = pd.DataFrame(
        {
            "sample_id": names,
            "true_label": np.asarray(labels, dtype=int),
            "predicted_label": (np.asarray(probs) >= threshold).astype(int),
            "alk_probability": np.asarray(probs, dtype=float),
        }
    )
    return metrics, df


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    train_ds = NiftiROIDataset(
        args.train_metadata, args.roi_root, args.train_split_name, build_ct_transform(train=True)
    )
    val_ds = NiftiROIDataset(
        args.val_metadata, args.roi_root, args.val_split_name, build_ct_transform(train=False)
    )
    external_ds = (
        NiftiROIDataset(
            args.external_metadata,
            args.roi_root,
            args.external_split_name,
            build_ct_transform(train=False),
        )
        if args.external_metadata
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    external_loader = (
        DataLoader(
            external_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if external_ds is not None
        else None
    )

    model = CTResNet18().to(device)
    if args.pretrained_weights:
        load_med3d_pretrained_weights(model, args.pretrained_weights, map_location=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_auc = -1.0
    best_path = output_dir / "best_ct_resnet18.pth"
    history = []

    print(f"{'epoch':>5} | {'train_loss':>10} | {'val_auc':>7} | {'val_acc':>7} | {'lr':>10}")
    print("-" * 52)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, y, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        val_metrics, _ = predict(model, val_loader, device, threshold=args.threshold)
        avg_loss = total_loss / max(len(train_loader), 1)
        row = {"epoch": epoch, "train_loss": avg_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"{epoch:5d} | {avg_loss:10.4f} | {val_metrics['auc']:7.4f} | {val_metrics['accuracy']:7.4f} | {args.lr:10.6f}")

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    split_outputs = []
    for split_name, loader in (("training", train_loader), ("validation", val_loader), ("external", external_loader)):
        if loader is None:
            continue
        metrics, pred_df = predict(model, loader, device, threshold=args.threshold)
        pred_df.insert(0, "split", split_name)
        split_outputs.append(pred_df)
        save_json(metrics, output_dir / f"metrics_{split_name}.json")
        print(f"{split_name}: AUC={metrics['auc']:.4f}, accuracy={metrics['accuracy']:.4f}")

    write_table(pd.concat(split_outputs, ignore_index=True), output_dir / "ct_predictions.csv")
    write_table(pd.DataFrame(history), output_dir / "training_history.csv")
    print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()

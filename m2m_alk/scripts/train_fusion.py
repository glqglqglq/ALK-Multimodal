from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from m2m_alk.data import MultiModalFeatureDataset, fit_transform_ct_features
from m2m_alk.fusion_models import build_fusion_model
from m2m_alk.utils import (
    binary_classification_metrics,
    ensure_dir,
    get_device,
    read_table,
    save_json,
    set_seed,
    write_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train patient-level multimodal fusion models.")
    parser.add_argument("--model", default="attention", choices=["concat", "gated", "attention", "m2m_alk"])
    parser.add_argument("--train-csv", required=True, help="CSV/XLSX: sample_id,wsi_feature_path,label,ct_features...")
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--external-csv", default=None)
    parser.add_argument("--output-dir", default="outputs/fusion_attention")
    parser.add_argument("--wsi-dim", type=int, default=512)
    parser.add_argument("--ct-dim", type=int, default=None, help="Auto-detected from table when omitted.")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--first-ct-feature-col", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def predict(model, loader, device, threshold: float) -> tuple[dict, pd.DataFrame]:
    model.eval()
    names, probs, labels = [], [], []
    with torch.no_grad():
        for wsi, ct, y, sample_ids in loader:
            logits = model(wsi.to(device), ct.to(device))
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            names.extend(sample_ids)
            probs.extend(p)
            labels.extend(y.numpy())
    metrics = binary_classification_metrics(labels, probs, threshold=threshold)
    pred_df = pd.DataFrame(
        {
            "sample_id": names,
            "true_label": np.asarray(labels, dtype=int),
            "predicted_label": (np.asarray(probs) >= threshold).astype(int),
            "alk_probability": np.asarray(probs, dtype=float),
        }
    )
    return metrics, pred_df


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    train_df = read_table(args.train_csv)
    val_df = read_table(args.val_csv)
    external_df = read_table(args.external_csv) if args.external_csv else None

    scaler, (train_df, val_df, external_df) = fit_transform_ct_features(
        train_df, val_df, external_df, first_ct_feature_col=args.first_ct_feature_col
    )
    joblib.dump(scaler, output_dir / "ct_feature_standard_scaler.joblib")

    ct_dim = args.ct_dim or (train_df.shape[1] - args.first_ct_feature_col)
    model = build_fusion_model(
        args.model,
        wsi_dim=args.wsi_dim,
        ct_dim=ct_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    train_ds = MultiModalFeatureDataset(train_df, first_ct_feature_col=args.first_ct_feature_col)
    val_ds = MultiModalFeatureDataset(val_df, first_ct_feature_col=args.first_ct_feature_col)
    external_ds = (
        MultiModalFeatureDataset(external_df, first_ct_feature_col=args.first_ct_feature_col)
        if external_df is not None
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    external_loader = (
        DataLoader(external_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
        if external_ds is not None
        else None
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_auc = -1.0
    no_improve = 0
    best_path = output_dir / f"best_{args.model}.pth"
    history = []

    print(f"{'epoch':>5} | {'train_loss':>10} | {'val_auc':>7} | {'val_acc':>7} | {'lr':>10}")
    print("-" * 52)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for wsi, ct, y, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(wsi.to(device), ct.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        scheduler.step()

        val_metrics, _ = predict(model, val_loader, device, threshold=args.threshold)
        avg_loss = total_loss / max(len(train_loader), 1)
        curr_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": avg_loss, **{f"val_{k}": v for k, v in val_metrics.items()}, "lr": curr_lr})
        print(f"{epoch:5d} | {avg_loss:10.4f} | {val_metrics['auc']:7.4f} | {val_metrics['accuracy']:7.4f} | {curr_lr:10.6f}")

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping after {args.patience} epochs without validation AUC improvement.")
                break

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

    write_table(pd.concat(split_outputs, ignore_index=True), output_dir / "fusion_predictions.csv")
    write_table(pd.DataFrame(history), output_dir / "training_history.csv")
    save_json(vars(args), output_dir / "run_args.json")
    print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()

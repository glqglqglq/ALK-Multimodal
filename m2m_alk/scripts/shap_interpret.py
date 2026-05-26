from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from torch.utils.data import DataLoader

from m2m_alk.data import MultiModalFeatureDataset, fit_transform_ct_features
from m2m_alk.fusion_models import build_fusion_model
from m2m_alk.utils import ensure_dir, get_device, read_table, write_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHAP interpretation for M2M-ALK fusion model.")
    parser.add_argument("--train-csv", required=True, help="Training table used as SHAP background reference.")
    parser.add_argument("--eval-csv", required=True, help="Evaluation table to explain.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="attention", choices=["attention", "m2m_alk", "concat", "gated"])
    parser.add_argument("--output-dir", default="outputs/shap")
    parser.add_argument("--wsi-dim", type=int, default=512)
    parser.add_argument("--ct-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--first-ct-feature-col", type=int, default=3)
    parser.add_argument("--background-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _collect_batch(dataset: MultiModalFeatureDataset, batch_size: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return next(iter(loader))


def _positive_class_shap(shap_values):
    """Return SHAP arrays for [wsi, ct] inputs and positive class."""
    # Older SHAP: list[class][input]
    if isinstance(shap_values, list) and shap_values and isinstance(shap_values[0], list):
        return np.asarray(shap_values[1][0]), np.asarray(shap_values[1][1])
    # Newer SHAP with multi-output arrays: list[input], each (n, features, classes)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        wsi_vals, ct_vals = np.asarray(shap_values[0]), np.asarray(shap_values[1])
        if wsi_vals.ndim == 3:
            wsi_vals = wsi_vals[:, :, 1]
        if ct_vals.ndim == 3:
            ct_vals = ct_vals[:, :, 1]
        return wsi_vals, ct_vals
    raise TypeError("Unsupported SHAP output structure.")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    train_df = read_table(args.train_csv)
    eval_df = read_table(args.eval_csv)
    _, (train_df, eval_df, _) = fit_transform_ct_features(
        train_df, eval_df, None, first_ct_feature_col=args.first_ct_feature_col
    )

    ct_dim = args.ct_dim or (train_df.shape[1] - args.first_ct_feature_col)
    model = build_fusion_model(
        args.model,
        wsi_dim=args.wsi_dim,
        ct_dim=ct_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    background_df = train_df.sample(
        n=min(args.background_size, len(train_df)), random_state=42, replace=False
    )
    bg_ds = MultiModalFeatureDataset(background_df, first_ct_feature_col=args.first_ct_feature_col)
    eval_ds = MultiModalFeatureDataset(eval_df, first_ct_feature_col=args.first_ct_feature_col)

    bg_wsi, bg_ct, _, _ = _collect_batch(bg_ds, len(bg_ds))
    eval_wsi, eval_ct, eval_labels, eval_names = _collect_batch(eval_ds, min(args.batch_size, len(eval_ds)))

    explainer = shap.GradientExplainer(model, [bg_wsi.to(device), bg_ct.to(device)])
    print("Computing SHAP values for the ALK-positive output...")
    shap_values = explainer.shap_values([eval_wsi.to(device), eval_ct.to(device)])
    wsi_shap, ct_shap = _positive_class_shap(shap_values)

    wsi_sample_contrib = np.abs(wsi_shap).mean(axis=1)
    ct_sample_contrib = np.abs(ct_shap).mean(axis=1)
    global_wsi = float(wsi_sample_contrib.mean())
    global_ct = float(ct_sample_contrib.mean())
    total = global_wsi + global_ct + 1e-12
    attribution_df = pd.DataFrame(
        {
            "modality": ["Pathology (WSI)", "Radiology (CT)"],
            "mean_abs_shap": [global_wsi, global_ct],
            "relative_attribution": [global_wsi / total, global_ct / total],
        }
    )
    write_table(attribution_df, output_dir / "modality_attribution.csv")

    plt.figure(figsize=(7, 7))
    plt.pie(
        attribution_df["relative_attribution"].values,
        labels=attribution_df["modality"].values,
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.title("M2M-ALK modality attribution")
    plt.savefig(output_dir / "shap_modality_attribution.png", dpi=600, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        ct_shap,
        eval_ct.cpu().numpy(),
        feature_names=[f"CT_Feat_{i}" for i in range(ct_shap.shape[1])],
        max_display=10,
        show=False,
        plot_type="dot",
    )
    plt.title("Radiology feature importance")
    plt.savefig(output_dir / "shap_radiology_beeswarm.png", dpi=600, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        wsi_shap,
        eval_wsi.cpu().numpy(),
        feature_names=[f"WSI_Feat_{i}" for i in range(wsi_shap.shape[1])],
        max_display=10,
        show=False,
        plot_type="dot",
    )
    plt.title("Pathology feature importance")
    plt.savefig(output_dir / "shap_pathology_beeswarm.png", dpi=600, bbox_inches="tight")
    plt.close()

    sample_df = pd.DataFrame(
        {
            "sample_id": list(eval_names),
            "true_label": eval_labels.numpy().astype(int),
            "pathology_mean_abs_shap": wsi_sample_contrib,
            "radiology_mean_abs_shap": ct_sample_contrib,
        }
    )
    write_table(sample_df, output_dir / "sample_level_shap_contribution.csv")
    print(f"Saved SHAP outputs to {output_dir}")


if __name__ == "__main__":
    main()

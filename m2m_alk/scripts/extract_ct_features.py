from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from m2m_alk.ct_model import CTResNet18
from m2m_alk.data import NiftiROIDataset, build_ct_transform
from m2m_alk.utils import get_device, write_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 512-dimensional CT features from a trained CT model.")
    parser.add_argument("--roi-root", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--split-name", required=True, help="Folder under roi-root, e.g., training/val/external.")
    parser.add_argument("--checkpoint", required=True, help="Trained CTResNet18 checkpoint.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    ds = NiftiROIDataset(args.metadata, args.roi_root, args.split_name, build_ct_transform(train=False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = CTResNet18().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    rows = []
    with torch.no_grad():
        for images, labels, names in loader:
            _, features = model(images.to(device), return_features=True)
            features_np = features.cpu().numpy()
            for sample_id, label, feat in zip(names, labels.numpy(), features_np):
                row = {"sample_id": sample_id, "label": int(label)}
                row.update({f"ct_feat_{i}": float(v) for i, v in enumerate(feat)})
                rows.append(row)

    write_table(pd.DataFrame(rows), args.output_csv)
    print(f"Saved CT features to {args.output_csv}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Return a torch.device from a CLI string."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Excel table."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write a CSV or Excel table based on filename extension."""
    path = Path(path)
    ensure_dir(path.parent)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    elif suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df.to_csv(path, index=False, sep=sep)
    else:
        raise ValueError(f"Unsupported output table format: {path}")


def torch_load_tensor(path: str | Path, map_location: str | torch.device = "cpu") -> torch.Tensor:
    """Load a tensor-like feature file using PyTorch's safer weights-only loader."""
    try:
        obj = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # older PyTorch versions do not expose weights_only
        obj = torch.load(path, map_location=map_location)

    if isinstance(obj, torch.Tensor):
        tensor = obj
    elif isinstance(obj, dict):
        tensor_values = [v for v in obj.values() if isinstance(v, torch.Tensor)]
        if len(tensor_values) != 1:
            raise ValueError(
                f"Expected one tensor in {path}, found {len(tensor_values)} tensor values."
            )
        tensor = tensor_values[0]
    else:
        raise TypeError(f"Unsupported feature object in {path}: {type(obj)!r}")
    return tensor.squeeze().float()


def binary_classification_metrics(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute patient-level binary metrics used by the manuscript."""
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_prob_arr = np.asarray(list(y_prob), dtype=float)
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision_ppv": float(
            precision_score(y_true_arr, y_pred_arr, zero_division=0)
        ),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        metrics["auc"] = float("nan")

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=labels).ravel()
    metrics.update(
        {
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
            "npv": float(tn / (tn + fn)) if (tn + fn) else float("nan"),
            "tp": float(tp),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
        }
    )
    return metrics


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_checkpoint_state_dict(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, torch.Tensor]:
    """Load a state_dict from common checkpoint formats."""
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint {path} does not contain a state_dict-like object.")
    return {str(k).replace("module.", ""): v for k, v in ckpt.items()}

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from m2m_alk.ct_model import CTResNet18
from m2m_alk.data import NiftiROIDataset, build_ct_display_transform, build_ct_transform
from m2m_alk.utils import ensure_dir, get_device


class GradCAM3D:
    """Grad-CAM for the CT stream using hooks on the final convolutional block."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def __call__(self, input_tensor: torch.Tensor, target_class: int | None = None) -> tuple[np.ndarray, int]:
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())
        self.model.zero_grad(set_to_none=True)
        output[0, target_class].backward()
        weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="trilinear", align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return cam_np, target_class

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CT Grad-CAM overlays.")
    parser.add_argument("--roi-root", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/ct_gradcam")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    output_dir = ensure_dir(Path(args.output_dir) / args.split_name)

    ds = NiftiROIDataset(
        args.metadata,
        args.roi_root,
        args.split_name,
        transform=build_ct_transform(train=False),
        return_display=True,
        display_transform=build_ct_display_transform(),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = CTResNet18().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    target_layer = model.backbone[-1]
    gradcam = GradCAM3D(model, target_layer)

    for idx, (img_model, img_display, _label, names) in enumerate(loader, start=1):
        if img_model.shape[0] != 1:
            raise ValueError("Grad-CAM export expects batch-size=1.")
        cam_3d, _ = gradcam(img_model.to(device))
        display_np = img_display.squeeze().numpy()
        slice_idx = int(np.unravel_index(np.argmax(cam_3d), cam_3d.shape)[0])
        sample_id = str(names[0])

        plt.figure(figsize=(4, 4))
        plt.imshow(display_np[slice_idx], cmap="gray")
        plt.axis("off")
        plt.savefig(output_dir / f"{sample_id}_raw.png", bbox_inches="tight", pad_inches=0, dpi=args.dpi)
        plt.close()

        plt.figure(figsize=(4, 4))
        plt.imshow(display_np[slice_idx], cmap="gray")
        plt.imshow(cam_3d[slice_idx], cmap="jet", alpha=0.45)
        plt.axis("off")
        plt.savefig(output_dir / f"{sample_id}_cam.png", bbox_inches="tight", pad_inches=0, dpi=args.dpi)
        plt.close()

        if idx % 50 == 0:
            print(f"Exported {idx}/{len(loader)} cases")

    gradcam.remove()
    print(f"Grad-CAM images saved to {output_dir}")


if __name__ == "__main__":
    main()

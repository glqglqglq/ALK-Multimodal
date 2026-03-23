import torch
import torch.nn as nn
from monai.networks.nets import ResNet
import os

class RadiologyModel(nn.Module):
    """
    Radiology branch using 3D ResNet-18. 
    Pretrained weights are sourced from Tencent MedicalNet (MedicalNet-Resnet18).
    """
    def __init__(self, weights_path=None, device='cuda'):
        super(RadiologyModel, self).__init__()
        # Initialize standard 3D ResNet18 via MONAI (architecture only)
        base = ResNet(
            block="basic", 
            layers=[2, 2, 2, 2], 
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3, 
            n_input_channels=1, 
            num_classes=2
        )

        # Load Tencent MedicalNet pretrained weights
        if weights_path and os.path.exists(weights_path):
            print(f"Loading Radiology pretrained weights from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location=device)
            # MedicalNet weights often contain 'module.' prefix and task-specific FC layers
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "") # Remove DataParallel prefix
                if "fc" not in name:            # Exclude final classification layers
                    new_state_dict[name] = v
            base.load_state_dict(new_state_dict, strict=False)
        
        # Use as feature extractor: exclude the final FC layers
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.maxpool = nn.AdaptiveMaxPool3d(1)

    def forward(self, x):
        x = self.backbone(x)
        # Global Feature Aggregation (1024-d)
        avg_x = self.avgpool(x)
        max_x = self.maxpool(x)
        feat = torch.cat([avg_x, max_x], dim=1)
        return feat.view(feat.size(0), -1)

class ALKFusionNet(nn.Module):
    """
    Multimodal framework integrating Pathology (WSI) and Radiology (CT) features.
    """
    def __init__(self, strategy='cross_attention', wsi_dim=512, rad_dim=1024, hidden_dim=128):
        super(ALKFusionNet, self).__init__()
        self.strategy = strategy
        
        # Projection heads to shared latent space
        self.wsi_proj = nn.Sequential(nn.Linear(wsi_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.rad_proj = nn.Sequential(nn.Linear(rad_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())

        if strategy == 'gated':
            self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.Sigmoid())
        elif strategy == 'cross_attention':
            self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, wsi_feat, rad_feat):
        h_w = self.wsi_proj(wsi_feat)
        h_r = self.rad_proj(rad_feat)

        if self.strategy == 'concat':
            z = torch.cat((h_w, h_r), dim=1)
        elif self.strategy == 'gated':
            combined = torch.cat((h_w, h_r), dim=1)
            z = combined * self.gate(combined)
        elif self.strategy == 'cross_attention':
            # WSI-led Cross-Attention
            q = h_w.unsqueeze(1)
            k = v = h_r.unsqueeze(1)
            attn_out, _ = self.attn(q, k, v)
            # Residual connection
            z_wsi = self.norm(attn_out.squeeze(1) + h_w)
            z = torch.cat((z_wsi, h_r), dim=1)

        return self.classifier(z)
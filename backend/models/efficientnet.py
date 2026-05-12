import os
import torch
import torch.nn as nn
import timm


class EfficientNetB4Detector(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,       # remove original head
            global_pool="avg",
        )
        in_features = self.backbone.num_features  # 1792
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)


def load_efficientnet(weights_path: str | None = None,
                      device: str = "cpu") -> EfficientNetB4Detector:
    print("  📥 Loading EfficientNet-B4 ...")
    model = EfficientNetB4Detector(num_classes=2, pretrained=True)
    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        print("  ✅ EfficientNet-B4 — custom weights loaded")
    else:
        print("  ⚠️  EfficientNet-B4 — ImageNet pretrained (no fine-tuned weights)")
    model.to(device).eval()
    return model


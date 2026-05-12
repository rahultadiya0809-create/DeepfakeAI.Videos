import os
import torch
import torch.nn as nn
import torchvision.models as tvm


class LipForensicsDetector(nn.Module):
    """
    Lightweight LipForensics-inspired model.
    Uses ResNet-50 backbone; in full LipForensics the backbone is a
    spatiotemporal ResNet trained on lip-reading, but here we adapt
    ResNet-50 for per-frame analysis with a mouth-region crop.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet  = tvm.resnet50(weights=weights)

        # Remove final FC layer — keep everything up to avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # (B, 2048, 1, 1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(2048),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)          # (B, 2048, 1, 1)
        return self.head(feat)


def load_lipforensics(weights_path: str | None = None,
                      device: str = "cpu") -> LipForensicsDetector:
    print("  📥 Loading LipForensics detector ...")
    model = LipForensicsDetector(num_classes=2, pretrained=True)
    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        print("  ✅ LipForensics — custom weights loaded")
    else:
        print("  ⚠️  LipForensics — ImageNet pretrained (no fine-tuned weights)")
    model.to(device).eval()
    return model
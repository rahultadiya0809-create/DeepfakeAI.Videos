import os
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel


# ─────────────────────────────────────────────────────────────
# Shared base class
# ─────────────────────────────────────────────────────────────
class SigLIPDetector(nn.Module):
    """Generic SigLIP vision-encoder-based detector."""

    def __init__(self, model_id: str, num_classes: int = 2, freeze_encoder: bool = True):
        super().__init__()
        print(f"    Downloading {model_id} (first run only) ...")
        full_model      = AutoModel.from_pretrained(model_id)
        self.encoder    = full_model.vision_model   # SigLIP vision encoder
        self.processor  = AutoProcessor.from_pretrained(model_id)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Hidden size: base=768, large=1024
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(0.3),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out    = self.encoder(pixel_values=pixel_values)
        pooled = out.pooler_output          # (B, hidden_size)
        return self.head(pooled)


# ─────────────────────────────────────────────────────────────
# SigLIP v1
# ─────────────────────────────────────────────────────────────
class SigLIPV1Detector(SigLIPDetector):
    MODEL_ID = "google/siglip-base-patch16-224"

    def __init__(self, num_classes: int = 2, freeze_encoder: bool = True):
        super().__init__(self.MODEL_ID, num_classes, freeze_encoder)


def load_siglip_v1(weights_path: str | None = None,
                   device: str = "cpu") -> SigLIPV1Detector:
    print("  📥 Loading SigLIP v1 ...")
    model = SigLIPV1Detector(num_classes=2, freeze_encoder=True)
    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        print("  ✅ SigLIP v1 — custom weights loaded")
    else:
        print("  ⚠️  SigLIP v1 — HuggingFace pretrained (no fine-tuned weights)")
    model.to(device).eval()
    return model


# ─────────────────────────────────────────────────────────────
# SigLIP 2
# ─────────────────────────────────────────────────────────────
class SigLIP2Detector(SigLIPDetector):
    MODEL_ID = "google/siglip2-base-patch16-224"

    def __init__(self, num_classes: int = 2, freeze_encoder: bool = True):
        super().__init__(self.MODEL_ID, num_classes, freeze_encoder)


def load_siglip2(weights_path: str | None = None,
                 device: str = "cpu") -> SigLIP2Detector:
    print("  📥 Loading SigLIP 2 ...")
    model = SigLIP2Detector(num_classes=2, freeze_encoder=True)
    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        print("  ✅ SigLIP 2 — custom weights loaded")
    else:
        print("  ⚠️  SigLIP 2 — HuggingFace pretrained (no fine-tuned weights)")
    model.to(device).eval()
    return model
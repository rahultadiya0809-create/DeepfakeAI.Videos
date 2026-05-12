import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleDetector(nn.Module):
    def __init__(self,
                 efficientnet,
                 lipforensics,
                 siglip_v1,
                 siglip2,
                 weights=(0.30, 0.25, 0.25, 0.20)):
        super().__init__()
        self.efficientnet = efficientnet
        self.lipforensics = lipforensics
        self.siglip_v1    = siglip_v1
        self.siglip2      = siglip2
        assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.w = weights

    @torch.no_grad()
    def forward(self,
                x_std:  torch.Tensor,   # (B,3,224,224) — standard norm
                x_sig:  torch.Tensor,   # (B,3,224,224) — SigLIP norm
                ) -> torch.Tensor:
        """
        Returns softmax probability tensor of shape (B, 2).
        Index 0 = REAL, Index 1 = FAKE.
        """
        p_eff  = F.softmax(self.efficientnet(x_std), dim=1)
        p_lip  = F.softmax(self.lipforensics(x_std), dim=1)
        p_sig1 = F.softmax(self.siglip_v1(x_sig),   dim=1)
        p_sig2 = F.softmax(self.siglip2(x_sig),      dim=1)

        combined = (
            self.w[0] * p_eff  +
            self.w[1] * p_lip  +
            self.w[2] * p_sig1 +
            self.w[3] * p_sig2
        )
        return combined  # (B, 2)

    @torch.no_grad()
    def forward_detailed(self,
                         x_std: torch.Tensor,
                         x_sig: torch.Tensor) -> dict:
        """Returns per-model scores + ensemble result.
        Supports both single-image (B=1) and batched (B>1) inputs.
        For batched: returns mean scores across the batch.
        """
        p_eff  = F.softmax(self.efficientnet(x_std), dim=1)
        p_lip  = F.softmax(self.lipforensics(x_std), dim=1)
        p_sig1 = F.softmax(self.siglip_v1(x_sig),   dim=1)
        p_sig2 = F.softmax(self.siglip2(x_sig),      dim=1)

        combined = (
            self.w[0] * p_eff  +
            self.w[1] * p_lip  +
            self.w[2] * p_sig1 +
            self.w[3] * p_sig2
        )

        # Mean across batch dimension for aggregated scores
        def mean_fake(t): return round(t[:, 1].mean().item(), 4)

        return {
            "efficientnet_b4": mean_fake(p_eff),
            "lipforensics":    mean_fake(p_lip),
            "siglip_v1":       mean_fake(p_sig1),
            "siglip2":         mean_fake(p_sig2),
            "ensemble":        mean_fake(combined),
        }
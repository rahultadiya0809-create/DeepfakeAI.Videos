import os
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from models.efficientnet  import load_efficientnet
from models.lipforensics  import load_lipforensics
from models.siglip        import load_siglip_v1, load_siglip2
from models.ensemble      import EnsembleDetector
from face_utils           import extract_faces_from_video, extract_face, extract_mouth_region
from frequency            import combined_frequency_score
from transforms           import STANDARD_TRANSFORM, SIGLIP_TRANSFORM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Weight paths ──────────────────────────────────────────────
WEIGHTS = {
    "efficientnet_b4": os.path.join(os.path.dirname(__file__), "models", "efficientnet_b4.pth"),
    "lipforensics":    os.path.join(os.path.dirname(__file__), "models", "lipforensics.pth"),
    "siglip_v1":       os.path.join(os.path.dirname(__file__), "models", "siglip_v1.pth"),
    "siglip2":         os.path.join(os.path.dirname(__file__), "models", "siglip2.pth"),
}


def load_all_models() -> EnsembleDetector:
    print(f"🔄 Device: {DEVICE}")
    print("🔄 Loading all 4 models ...")

    eff  = load_efficientnet(WEIGHTS["efficientnet_b4"], str(DEVICE))
    lip  = load_lipforensics(WEIGHTS["lipforensics"],    str(DEVICE))
    sig1 = load_siglip_v1(  WEIGHTS["siglip_v1"],       str(DEVICE))
    sig2 = load_siglip2(    WEIGHTS["siglip2"],          str(DEVICE))

    ensemble = EnsembleDetector(eff, lip, sig1, sig2,
                                weights=(0.30, 0.25, 0.25, 0.20))
    print("✅ All models loaded and ready.")
    return ensemble


# ── Helpers ───────────────────────────────────────────────────
def _to_std(pil: Image.Image) -> torch.Tensor:
    return STANDARD_TRANSFORM(pil).unsqueeze(0).to(DEVICE)

def _to_sig(pil: Image.Image) -> torch.Tensor:
    return SIGLIP_TRANSFORM(pil).unsqueeze(0).to(DEVICE)


def _batch_transform(faces: list[Image.Image], transform) -> torch.Tensor:
    """Transform a list of PIL images into a single batched tensor."""
    tensors = [transform(f) for f in faces]
    return torch.stack(tensors).to(DEVICE)


# ── Single image prediction ───────────────────────────────────
def predict_image(image: Image.Image, ensemble: EnsembleDetector) -> dict:
    face  = extract_face(image)
    mouth = extract_mouth_region(image)
    if face is None:
        return {"error": "No face detected in the image"}

    x_std = _to_std(face)
    x_sig = _to_sig(face)

    scores = ensemble.forward_detailed(x_std, x_sig)
    fake_p = round(scores["ensemble"] * 100, 2)
    real_p = round((1 - scores["ensemble"]) * 100, 2)
    freq   = combined_frequency_score(np.array(face))

    return {
        "prediction":        "FAKE" if fake_p > 50 else "REAL",
        "fake_probability":  fake_p,
        "real_probability":  real_p,
        "fft_anomaly_score": freq,
        "frames_analyzed":   1,
        "model_scores": {
            "efficientnet_b4": round(scores["efficientnet_b4"], 4),
            "lipforensics":    round(scores["lipforensics"],    4),
            "siglip_v1":       round(scores["siglip_v1"],       4),
            "siglip2":         round(scores["siglip2"],         4),
        },
    }


# ── Video prediction (OPTIMISED — batched inference) ──────────
def predict_video(video_path: str, ensemble: EnsembleDetector) -> dict:
    frames = extract_faces_from_video(video_path, max_frames=15)
    if not frames:
        return {"error": "No faces detected in the video"}

    # --- Batch all face crops into single tensors ---
    faces = [f["face"] for f in frames]
    batch_std = _batch_transform(faces, STANDARD_TRANSFORM)   # (B, 3, 224, 224)
    batch_sig = _batch_transform(faces, SIGLIP_TRANSFORM)     # (B, 3, 224, 224)

    # --- ONE call to all 4 models with full batch ---
    scores = ensemble.forward_detailed(batch_std, batch_sig)

    # --- FFT on SUBSAMPLE only (5 evenly-spaced frames) ---
    fft_indices = np.linspace(0, len(faces) - 1, min(5, len(faces)), dtype=int)
    fft_scores  = [combined_frequency_score(np.array(faces[i])) for i in fft_indices]

    fake_p = round(scores["ensemble"] * 100, 2)
    real_p = round((1 - scores["ensemble"]) * 100, 2)

    return {
        "prediction":        "FAKE" if fake_p > 50 else "REAL",
        "fake_probability":  fake_p,
        "real_probability":  real_p,
        "fft_anomaly_score": round(float(np.mean(fft_scores)), 4),
        "frames_analyzed":   len(frames),
        "model_scores": {
            "efficientnet_b4": round(scores["efficientnet_b4"], 4),
            "lipforensics":    round(scores["lipforensics"],    4),
            "siglip_v1":       round(scores["siglip_v1"],       4),
            "siglip2":         round(scores["siglip2"],         4),
        },
    }
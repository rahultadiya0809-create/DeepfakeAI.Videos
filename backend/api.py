import io
import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from detector import load_all_models, predict_image, predict_video

app = FastAPI(
    title="DeepScan AI — Deepfake Detection API",
    version="2.0.0",
    description="4-model ensemble: EfficientNet-B4 + LipForensics + SigLIP v1 + SigLIP 2",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models at startup ────────────────────────────────────
print("=" * 55)
ensemble = load_all_models()
print("=" * 55)


# ── Routes ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": [
            "EfficientNet-B4",
            "LipForensics",
            "SigLIP v1",
            "SigLIP 2",
        ],
        "ensemble_weights": {
            "efficientnet_b4": "30%",
            "lipforensics":    "25%",
            "siglip_v1":       "25%",
            "siglip2":         "20%",
        },
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    content_type = (file.content_type or "").lower()
    fname        = (file.filename or "upload").lower()

    # ── Image ─────────────────────────────────────────────────
    if content_type.startswith("image/") or fname.endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    ):
        data = await file.read()
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            raise HTTPException(400, "Cannot open image file")
        result = predict_image(image, ensemble)
        if "error" in result:
            raise HTTPException(422, result["error"])
        return result

    # ── Video ─────────────────────────────────────────────────
    elif content_type.startswith("video/") or fname.endswith(
        (".mp4", ".avi", ".mov", ".mkv", ".webm", ".3gp")
    ):
        suffix = os.path.splitext(fname)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        try:
            result = predict_video(tmp_path, ensemble)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        if "error" in result:
            raise HTTPException(422, result["error"])
        return result

    else:
        raise HTTPException(
            415,
            f"Unsupported file type '{content_type}'. Upload an image or video.",
        )
import os
import cv2
import numpy as np
from PIL import Image
import torch

try:
    from facenet_pytorch import MTCNN
    _MTCNN_AVAILABLE = True
except ImportError:
    _MTCNN_AVAILABLE = False
    print("⚠️  facenet-pytorch not installed — face detection disabled")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialise MTCNN once globally
_mtcnn = None

def _get_mtcnn():
    global _mtcnn
    if _mtcnn is None and _MTCNN_AVAILABLE:
        _mtcnn = MTCNN(
            image_size=224,
            margin=30,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            keep_all=False,
            device=DEVICE,
        )
    return _mtcnn


def _detect_face_box(image: Image.Image):
    """Run MTCNN once and return the best bounding box, or None."""
    mtcnn = _get_mtcnn()
    if mtcnn is None:
        return None
    boxes, probs = mtcnn.detect(image)
    if boxes is None or len(boxes) == 0:
        return None
    best = int(np.argmax(probs))
    return [int(v) for v in boxes[best]]


def _crop_face(image: Image.Image, box) -> Image.Image | None:
    """Crop face from image using a bounding box [x1, y1, x2, y2]."""
    arr = np.array(image)
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = arr[y1:y2, x1:x2]
    return Image.fromarray(crop).resize((224, 224), Image.LANCZOS)


def _crop_mouth(face: Image.Image) -> Image.Image:
    """Crop the lower 45% of a face image (mouth/lip region)."""
    arr = np.array(face)
    h = arr.shape[0]
    mouth = arr[int(h * 0.55):h, :, :]
    return Image.fromarray(mouth).resize((224, 224), Image.LANCZOS)


def extract_face(image: Image.Image) -> Image.Image | None:
    """
    Detect the largest/most-confident face in a PIL image.
    Returns a 224×224 PIL Image or None if no face found.
    """
    box = _detect_face_box(image)
    if box is None:
        mtcnn = _get_mtcnn()
        if mtcnn is None:
            # Fallback: just resize the whole image
            return image.resize((224, 224), Image.LANCZOS)
        return None
    return _crop_face(image, box)


def extract_mouth_region(image: Image.Image) -> Image.Image:
    """
    Crop the lower-third of the face (mouth/lip region) for LipForensics.
    Falls back to a centred crop if face not detected.
    """
    face = extract_face(image)
    if face is None:
        face = image.resize((224, 224), Image.LANCZOS)
    return _crop_mouth(face)


def extract_face_and_mouth(image: Image.Image) -> tuple[Image.Image, Image.Image] | None:
    """
    Detect face ONCE and derive both face + mouth crops.
    Returns (face_224, mouth_224) or None if no face found.
    """
    box = _detect_face_box(image)
    if box is None:
        mtcnn = _get_mtcnn()
        if mtcnn is None:
            # Fallback: use whole image
            face = image.resize((224, 224), Image.LANCZOS)
            return face, _crop_mouth(face)
        return None

    face = _crop_face(image, box)
    if face is None:
        return None
    return face, _crop_mouth(face)


def extract_faces_from_video(video_path: str,
                              max_frames: int = 15) -> list[dict]:
    """
    Sample up to max_frames evenly-spaced frames from a video.
    Returns list of dicts with keys: 'face', 'mouth', 'frame_idx'.
    Uses single MTCNN call per frame (not two).
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
    results = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil  = Image.fromarray(rgb)
        pair = extract_face_and_mouth(pil)
        if pair is None:
            continue
        face, mouth = pair
        results.append({"face": face, "mouth": mouth, "frame_idx": int(idx)})

    cap.release()
    return results
import numpy as np
import cv2
from PIL import Image


def fft_score(face_arr: np.ndarray) -> float:
    """
    2-D FFT on grayscale face. Returns anomaly score 0-1.
    Higher → more high-frequency energy → likely synthetic.
    """
    if face_arr.ndim == 3:
        gray = cv2.cvtColor(face_arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = face_arr
    gray  = gray.astype(np.float32)
    mag   = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    h, w  = mag.shape
    cy, cx = h // 2, w // 2

    def ring(r0, r1):
        y, x = np.ogrid[:h, :w]
        d    = np.sqrt((x - cx)**2 + (y - cy)**2)
        m    = (d >= r0) & (d < r1)
        return float(mag[m].mean()) if m.sum() > 0 else 0.0

    low  = ring(0,           min(h,w)//8)
    high = ring(min(h,w)//4, min(h,w)//2)
    tot  = low + high + 1e-8
    return float(np.clip((high/tot - 0.3) / 0.4, 0.0, 1.0))


def dct_score(face_arr: np.ndarray) -> float:
    """
    8×8 block DCT — detects GAN block-level compression artefacts.
    Returns anomaly score 0-1.
    """
    if face_arr.ndim == 3:
        gray = cv2.cvtColor(face_arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = face_arr
    gray  = cv2.resize(gray.astype(np.float32), (128, 128))
    scores = []
    for i in range(0, 120, 8):
        for j in range(0, 120, 8):
            block = gray[i:i+8, j:j+8]
            d     = cv2.dct(block)
            scores.append(float(np.abs(d[4:, 4:]).mean()))
    if not scores:
        return 0.0
    return float(np.clip((np.mean(scores) * np.std(scores)) / 600.0, 0.0, 1.0))


def combined_frequency_score(face_arr: np.ndarray) -> float:
    """Weighted combination of FFT + DCT scores."""
    return round(0.6 * fft_score(face_arr) + 0.4 * dct_score(face_arr), 4)
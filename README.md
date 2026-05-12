# 🔍 DeepScan AI — Deepfake Detection

A 4-model ensemble deepfake detection system using **EfficientNet-B4**, **LipForensics**, **SigLIP v1**, and **SigLIP 2** to detect AI-generated or manipulated video content.

> 🎓 **College Project** — AI/ML Deepfake Detection

## ✨ Features

- **4-Model Ensemble** — Weighted voting: EfficientNet-B4 (30%) + LipForensics (25%) + SigLIP v1 (25%) + SigLIP 2 (20%)
- **Video Analysis** — Upload MP4/AVI/MOV/MKV/WebM videos for frame-by-frame deepfake detection
- **Image Analysis** — Single image deepfake detection
- **Face Detection** — MTCNN-based automatic face and lip-region cropping
- **FFT Frequency Analysis** — Detects GAN artifacts in the frequency domain
- **Beautiful UI** — Modern dark-theme frontend with real-time scanning animation
- **Demo Mode** — Works without backend for UI demonstration

## 🏗️ Architecture

```
DeepfakeAI/
├── forntend.html          # Frontend UI (single-page app)
├── train.py               # Training script for all 4 models
├── requriments.txt        # Python dependencies
├── backend/
│   ├── api.py             # FastAPI REST API
│   ├── detector.py        # Detection pipeline (batched inference)
│   ├── face_utils.py      # MTCNN face/lip extraction
│   ├── frequency.py       # FFT + DCT frequency analysis
│   ├── transforms.py      # Image preprocessing transforms
│   └── models/
│       ├── efficientnet.py  # EfficientNet-B4 detector
│       ├── lipforensics.py  # LipForensics detector (ResNet-50)
│       ├── siglip.py        # SigLIP v1 + SigLIP 2 detectors
│       └── ensemble.py      # Weighted ensemble combiner
└── datasets/              # Training data (not included)
```

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/DeepfakeAI.git
cd DeepfakeAI
```

### 2. Install dependencies
```bash
pip install -r requriments.txt
```

> ⚠️ For `facenet-pytorch`, if you face dependency issues, install separately:
> ```bash
> pip install facenet-pytorch --no-deps
> ```

### 3. Train the models (optional)
Place your dataset in `datasets/` folder with two subfolders: `real/` and `fake/`, then run:
```bash
python train.py
```
This trains all 4 models and saves `.pth` weights to `backend/models/`.

### 4. Start the backend server
```bash
cd backend
uvicorn api:app --reload --port 8000
```

### 5. Open the frontend
Open `forntend.html` in your browser. It will auto-detect the backend server.

## 🧠 Models

| Model | Type | Parameters | Weight | Description |
|-------|------|-----------|--------|-------------|
| EfficientNet-B4 | CNN | 19M | 30% | Detects GAN blending artifacts and texture anomalies |
| LipForensics | ResNet-50 | 25M | 25% | Lip-region temporal inconsistency detection |
| SigLIP v1 | ViT | 86M | 25% | Google's sigmoid image-language pre-trained model |
| SigLIP 2 | ViT | 93M | 20% | Enhanced multi-resolution vision transformer |

## 🛠️ Tech Stack

- **Backend:** Python, PyTorch, FastAPI
- **Models:** timm, HuggingFace Transformers
- **Face Detection:** MTCNN (facenet-pytorch)
- **Video Processing:** OpenCV
- **Frontend:** HTML, CSS, JavaScript

## 📝 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server health check & model status |
| `POST` | `/detect` | Upload image or video for deepfake detection |

## ⚠️ Disclaimer

This project is for **educational purposes only**. It is a college project demonstrating AI/ML deepfake detection techniques.

## 📄 License

This project is open source for educational use.

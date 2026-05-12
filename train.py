import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from models.efficientnet import EfficientNetB4Detector
from models.lipforensics  import LipForensicsDetector
from models.siglip        import SigLIPV1Detector, SigLIP2Detector

# ── Config ────────────────────────────────────────────────────
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS   = 10
BATCH    = 16        # reduce to 8 if GPU OOM
LR       = 1e-4
DATASET  = os.path.join(os.path.dirname(__file__), "..", "dataset")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "models")

# ── Transforms ────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
siglip_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ── Dataset loading ───────────────────────────────────────────
def get_loaders(transform):
    ds      = datasets.ImageFolder(DATASET, transform=transform)
    n       = len(ds)
    trn     = int(0.8 * n)
    val     = n - trn
    tr, va  = random_split(ds, [trn, val],
                           generator=torch.Generator().manual_seed(42))
    tr_l    = DataLoader(tr, batch_size=BATCH, shuffle=True,  num_workers=0)
    va_l    = DataLoader(va, batch_size=BATCH, shuffle=False, num_workers=0)
    return tr_l, va_l, trn, val


# ── Training loop ─────────────────────────────────────────────
def train_one(model, name, save_name, transform=None):
    print(f"\n{'='*55}")
    print(f" Training: {name}")
    print(f"{'='*55}")
    tf          = transform or train_tf
    tr_l, va_l, trn, valn = get_loaders(tf)
    print(f"  Train: {trn} | Val: {valn} | Device: {DEVICE}")

    model.to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    best_acc, best_path = 0.0, os.path.join(SAVE_DIR, save_name)

    for ep in range(EPOCHS):
        t0 = time.time()

        # Train
        model.train()
        t_loss = t_corr = 0
        for imgs, labels in tr_l:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            out  = model(imgs)
            loss = crit(out, labels)
            loss.backward()
            opt.step()
            t_loss += loss.item()
            t_corr += (out.argmax(1) == labels).sum().item()

        # Validate
        model.eval()
        v_corr = 0
        with torch.no_grad():
            for imgs, labels in va_l:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                v_corr += (model(imgs).argmax(1) == labels).sum().item()

        sched.step()
        t_acc = t_corr / trn * 100
        v_acc = v_corr / valn * 100
        print(f"  Ep {ep+1:02d}/{EPOCHS} | "
              f"Loss {t_loss/len(tr_l):.4f} | "
              f"Train {t_acc:.1f}% | "
              f"Val {v_acc:.1f}% | "
              f"{time.time()-t0:.1f}s")

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), best_path)
            print(f"  💾 Saved → {save_name}  (val {v_acc:.1f}%)")

    print(f"  ✅ {name} done. Best val acc: {best_acc:.1f}%")
    return best_acc


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Device : {DEVICE}")
    print(f"Dataset: {DATASET}")
    print(f"Save to: {SAVE_DIR}")

    results = {}

    results["EfficientNet-B4"] = train_one(
        EfficientNetB4Detector(pretrained=True),
        "EfficientNet-B4", "efficientnet_b4.pth", train_tf)

    results["LipForensics"] = train_one(
        LipForensicsDetector(pretrained=True),
        "LipForensics", "lipforensics.pth", train_tf)

    results["SigLIP v1"] = train_one(
        SigLIPV1Detector(freeze_encoder=True),
        "SigLIP v1", "siglip_v1.pth", siglip_tf)

    results["SigLIP 2"] = train_one(
        SigLIP2Detector(freeze_encoder=True),
        "SigLIP 2", "siglip2.pth", siglip_tf)

    print(f"\n{'='*55}")
    print(" TRAINING COMPLETE")
    print(f"{'='*55}")
    for m, acc in results.items():
        print(f"  {m:20s}  {acc:.1f}%")
    print(f"\n✅ All .pth files saved in backend/models/")
    print("🚀 Run: cd backend && uvicorn api:app --reload --port 8000")
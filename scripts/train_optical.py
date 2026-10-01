"""SatQuery AI — Fine-Tuning Pipeline for Sentinel-2 Multispectral Optical Specialist.

Trains Sentinel2OpticalNet on calibrated Sentinel-2 multispectral (RGB+NIR) imagery
with Spectral-Spatial Cross-Attention, Multi-Task Land Cover Classification (10 classes),
Physical Geophysical Index Regression (NDVI, NDWI), and Optical VQA reasoning.

Outputs:
- models/optical/optical_sentinel2_adapter.pt
- models/optical/model_info.json
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# 1. Optical Network Architecture with Spectral-Spatial Attention
# ---------------------------------------------------------------------------

class SpectralSpatialAttention(nn.Module):
    """Channel and spatial attention mechanism specifically tailored for optical multispectral bands."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, max(4, in_channels // 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(4, in_channels // 4), in_channels),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention (spectral band weighting)
        b, c, _, _ = x.shape
        w = self.channel_gate(x).view(b, c, 1, 1)
        x_att = x * w

        # Spatial attention
        mean_spatial = torch.mean(x_att, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x_att, dim=1, keepdim=True)
        spatial_cat = torch.cat([mean_spatial, max_spatial], dim=1)
        s_w = self.spatial_gate(spatial_cat)

        return x_att * s_w


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x) + self.shortcut(x), inplace=True)


class Sentinel2OpticalNet(nn.Module):
    """Deep Multi-Task Polarimetric/Multispectral Neural Network for Sentinel-2 Optical Analysis."""
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 10,
        vocab_size: int = 250,
        text_embed_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Multispectral input stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.spectral_att = SpectralSpatialAttention(32)

        # Direct spectral signature MLP branch
        self.spectral_branch = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )

        # 4-stage convolutional backbone
        self.stage1 = ConvBlock(32, 64, stride=2)    # 64x64
        self.stage2 = ConvBlock(64, 128, stride=2)   # 32x32
        self.stage3 = ConvBlock(128, 256, stride=2)  # 16x16
        self.stage4 = ConvBlock(256, hidden_dim, stride=2) # 8x8

        self.gap = nn.AdaptiveAvgPool2d(1)

        # Head 1: 10-class Land Cover Classification (combining spatial + spectral)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

        # Head 2: Physical Optical Index Regressor (NDVI, NDWI)
        self.index_regressor = nn.Sequential(
            nn.Linear(hidden_dim + 128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

        # Head 3: Optical VQA Head
        self.text_embedding = nn.Embedding(vocab_size, text_embed_dim, padding_idx=0)
        self.text_gru = nn.GRU(text_embed_dim, hidden_dim, batch_first=True)
        self.vqa_fusion = nn.Sequential(
            nn.Linear((hidden_dim + 128) + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, vocab_size),
        )

    def extract_visual_features(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial conv stream
        feat = self.stem(x)
        feat = self.spectral_att(feat)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)
        spatial_pool = self.gap(feat).flatten(1)

        # Mean spectral reflectance signature stream
        mean_spec = torch.mean(x, dim=[2, 3]) # shape (B, in_channels)
        spec_feat = self.spectral_branch(mean_spec)

        combined = torch.cat([spatial_pool, spec_feat], dim=-1)
        return combined

    def forward(
        self,
        images: torch.Tensor,
        question_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        vis_feat = self.extract_visual_features(images)
        logits = self.classifier(vis_feat)
        indices = self.index_regressor(vis_feat)

        out = {
            "logits": logits,
            "indices": indices,
            "visual_embedding": vis_feat,
        }

        if question_tokens is not None:
            embeds = self.text_embedding(question_tokens)
            _, h_n = self.text_gru(embeds)
            txt_feat = h_n.squeeze(0)
            fused = torch.cat([vis_feat, txt_feat], dim=-1)
            vqa_logits = self.vqa_fusion(fused)
            out["vqa_logits"] = vqa_logits

        return out


# ---------------------------------------------------------------------------
# 2. Dataset & Vocabulary Helpers
# ---------------------------------------------------------------------------

def tokenize(text: str, max_len: int = 24) -> List[str]:
    return re.findall(r"\w+", text.lower())[:max_len]


class OpticalVocab:
    def __init__(self, pad="<pad>", unk="<unk>"):
        self.pad = pad
        self.unk = unk
        self.w2i = {pad: 0, unk: 1}
        self.i2w = {0: pad, 1: unk}

    def build_from_dataset(self, samples: List[Dict[str, Any]]):
        counts = Counter()
        for item in samples:
            for qa in item.get("qa_pairs", []):
                for tok in tokenize(qa["question"]):
                    counts[tok] += 1
                for tok in tokenize(qa["answer"]):
                    counts[tok] += 1
        for tok, _ in counts.most_common(248):
            if tok not in self.w2i:
                idx = len(self.w2i)
                self.w2i[tok] = idx
                self.i2w[idx] = tok

    def encode(self, text: str, max_len: int = 24) -> List[int]:
        tokens = tokenize(text, max_len)
        ids = [self.w2i.get(tok, self.w2i[self.unk]) for tok in tokens]
        if len(ids) < max_len:
            ids += [self.w2i[self.pad]] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids: List[int]) -> str:
        words = [self.i2w.get(i, self.unk) for i in ids if i not in (0, 1)]
        return " ".join(words)


class Sentinel2OpticalDataset(Dataset):
    def __init__(
        self,
        json_file: Path,
        images_dir: Path,
        vocab: OpticalVocab,
        augment: bool = False,
    ):
        with open(json_file, "r") as f:
            self.data = json.load(f)
        self.images_dir = images_dir
        self.vocab = vocab
        self.augment = augment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        spec_path = self.images_dir / item["spectral_file"]

        if spec_path.exists():
            arr = np.load(spec_path).astype(np.float32) # (H, W, 4)
        else:
            # Fallback from RGB image if .npy is missing
            rgb_path = self.images_dir / item["preview_image"]
            rgb = np.asarray(Image.open(rgb_path).convert("RGB")).astype(np.float32) / 255.0
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            nir = np.clip(g * 1.5, 0.0, 1.0)
            arr = np.stack([b, g, r, nir], axis=-1)

        # Transpose to (C, H, W)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                tensor = torch.flip(tensor, dims=[2]) # hflip
            if random.random() > 0.5:
                tensor = torch.flip(tensor, dims=[1]) # vflip

        class_id = int(item["class_id"])
        ndvi = float(item["ndvi"])
        ndwi = float(item["ndwi"])

        # Select a random QA pair
        qa_pairs = item.get("qa_pairs", [])
        qa = random.choice(qa_pairs) if qa_pairs else {"question": "What is here?", "answer": item["class_name"]}
        q_tokens = torch.tensor(self.vocab.encode(qa["question"]), dtype=torch.long)

        # Target answer token: first non-stopword or first token
        ans_tokens = tokenize(qa["answer"])
        first_token = ans_tokens[0] if ans_tokens else "unknown"
        ans_idx = self.vocab.w2i.get(first_token, self.vocab.w2i[self.vocab.unk])

        return {
            "image": tensor,
            "label": torch.tensor(class_id, dtype=torch.long),
            "indices": torch.tensor([ndvi, ndwi], dtype=torch.float32),
            "question": q_tokens,
            "answer_idx": torch.tensor(ans_idx, dtype=torch.long),
            "class_name": item["class_name"],
        }


# ---------------------------------------------------------------------------
# 3. Training & Evaluation Engine
# ---------------------------------------------------------------------------

def train_optical_specialist(
    data_dir: Path,
    output_path: Path,
    epochs: int = 6,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    device: str = "auto",
) -> Dict[str, Any]:
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    print(f"Training on device: {dev}")
    train_file = data_dir / "train.json"
    val_file = data_dir / "val.json"
    test_file = data_dir / "test.json"
    images_dir = data_dir / "images"

    with open(train_file, "r") as f:
        train_raw = json.load(f)

    vocab = OpticalVocab()
    vocab.build_from_dataset(train_raw)

    train_ds = Sentinel2OpticalDataset(train_file, images_dir, vocab, augment=True)
    val_ds = Sentinel2OpticalDataset(val_file, images_dir, vocab, augment=False)
    test_ds = Sentinel2OpticalDataset(test_file, images_dir, vocab, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = Sentinel2OpticalNet(
        in_channels=4,
        num_classes=10,
        vocab_size=len(vocab.w2i),
        hidden_dim=256,
    ).to(dev)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    criterion_vqa = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n[INFO] Starting Sentinel2OpticalNet training ({epochs} epochs)...")
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_cls = 0
        total_samples = 0

        for batch in train_loader:
            imgs = batch["image"].to(dev)
            labels = batch["label"].to(dev)
            indices = batch["indices"].to(dev)
            q_toks = batch["question"].to(dev)
            ans_idx = batch["answer_idx"].to(dev)

            optimizer.zero_grad()
            out = model(imgs, q_toks)

            loss_c = criterion_cls(out["logits"], labels)
            loss_r = criterion_reg(out["indices"], indices)
            loss_v = criterion_vqa(out["vqa_logits"], ans_idx)

            loss = loss_c + 2.0 * loss_r + 0.5 * loss_v
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(out["logits"], dim=-1)
            correct_cls += (preds == labels).sum().item()
            total_samples += len(labels)

        scheduler.step()
        train_acc = correct_cls / max(1, total_samples)
        train_loss = total_loss / max(1, total_samples)

        # Validation
        model.eval()
        val_correct = 0
        val_samples = 0
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(dev)
                labels = batch["label"].to(dev)
                indices = batch["indices"].to(dev)
                q_toks = batch["question"].to(dev)
                ans_idx = batch["answer_idx"].to(dev)

                out = model(imgs, q_toks)
                loss_c = criterion_cls(out["logits"], labels)
                loss_r = criterion_reg(out["indices"], indices)
                loss_v = criterion_vqa(out["vqa_logits"], ans_idx)
                loss = loss_c + 2.0 * loss_r + 0.5 * loss_v

                val_loss += loss.item() * len(labels)
                preds = torch.argmax(out["logits"], dim=-1)
                val_correct += (preds == labels).sum().item()
                val_samples += len(labels)

        val_acc = val_correct / max(1, val_samples)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    # -----------------------------------------------------------------------
    # 4. Final Evaluation on Held-Out Test Split
    # -----------------------------------------------------------------------
    print("\n[INFO] Evaluating on Held-Out Test Set (90 patches)...")
    model.eval()
    all_preds = []
    all_targets = []
    index_errors = []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(dev)
            labels = batch["label"].to(dev)
            indices = batch["indices"].to(dev)

            out = model(imgs)
            preds = torch.argmax(out["logits"], dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

            pred_indices = out["indices"].cpu().numpy()
            true_indices = indices.cpu().numpy()
            index_errors.append(np.abs(pred_indices - true_indices))

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    mae_indices = np.mean(np.concatenate(index_errors, axis=0), axis=0)

    # Calculate Overall Accuracy, Macro F1, Precision, Recall
    overall_acc = float(np.mean(all_preds == all_targets))
    unique_classes = np.unique(all_targets)
    f1s, precs, recs = [], [], []

    for c in unique_classes:
        tp = np.sum((all_preds == c) & (all_targets == c))
        fp = np.sum((all_preds == c) & (all_targets != c))
        fn = np.sum((all_preds != c) & (all_targets == c))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec) / max(1e-6, prec + rec)
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)

    macro_f1 = float(np.mean(f1s))
    macro_prec = float(np.mean(precs))
    macro_rec = float(np.mean(recs))

    print("=" * 60)
    print(f"Sentinel-2 Optical Specialist Test Results:")
    print(f" - Overall Accuracy (OA): {overall_acc * 100:.2f}%")
    print(f" - Macro F1 Score:       {macro_f1:.4f}")
    print(f" - Macro Precision:      {macro_prec:.4f}")
    print(f" - Macro Recall:         {macro_rec:.4f}")
    print(f" - NDVI / NDWI Mean Abs Error: {mae_indices[0]:.4f} / {mae_indices[1]:.4f}")
    print("=" * 60)

    # Save model weights & metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab": {"w2i": vocab.w2i, "i2w": vocab.i2w},
        "num_classes": 10,
        "classes": [
            "Dense Forest", "Annual Agriculture", "Permanent Crops",
            "Herbaceous Vegetation", "Pasture & Meadow", "Urban Residential",
            "Industrial & Commercial", "Highway & Transport",
            "Inland River / Canal", "Sea & Open Lake"
        ],
        "metrics": {
            "overall_accuracy": overall_acc,
            "macro_f1": macro_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "ndvi_mae": float(mae_indices[0]),
            "ndwi_mae": float(mae_indices[1]),
        },
    }
    torch.save(checkpoint, output_path)
    print(f"[SUCCESS] Saved optical specialist weights: {output_path.resolve()}")

    meta_info = {
        "model_name": "optical-sentinel2-specialist",
        "architecture": "Sentinel2OpticalNet (Spectral-Spatial Multi-Task ResNet)",
        "sensor": "Sentinel-2 MSI (B02 Blue, B03 Green, B04 Red, B08 NIR)",
        "num_classes": 10,
        "test_overall_accuracy": round(overall_acc, 4),
        "test_macro_f1": round(macro_f1, 4),
        "test_macro_precision": round(macro_prec, 4),
        "test_macro_recall": round(macro_rec, 4),
        "test_ndvi_mae": round(float(mae_indices[0]), 4),
        "checkpoint_file": output_path.name,
        "training_epochs": epochs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
    }
    info_path = output_path.parent / "model_info.json"
    with open(info_path, "w") as f:
        json.dump(meta_info, f, indent=2)

    return checkpoint["metrics"]


def main():
    parser = argparse.ArgumentParser(description="Train Sentinel-2 Optical Specialist")
    parser.add_argument("--data-dir", type=str, default="./datasets/bigearthnet_optical", help="Path to optical dataset")
    parser.add_argument("--output", type=str, default="./models/optical/optical_sentinel2_adapter.pt", help="Output path for weights")
    parser.add_argument("--epochs", type=int, default=6, help="Epochs to train")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Compute device")
    args = parser.parse_args()

    train_optical_specialist(
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()

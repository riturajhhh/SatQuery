"""SatQuery AI — Sentinel-1 SAR Dedicated Model Training Pipeline.

Fine-tunes a specialized deep polarimetric neural network on BigEarthNet Sentinel-1 SAR data
for multi-label radar land-cover classification, dielectric double-bounce localization,
and radar question answering (SAR-VQA).

Outputs:
- models/optical_sar/sar_sentinel1_adapter.pt (Trained PyTorch weights & vocabularies)
- models/optical_sar/model_info.json (Model metadata, training parameters & evaluation metrics)
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


# ---------------------------------------------------------------------------
# Neural Architecture: Sentinel-1 Polarimetric SAR Network
# ---------------------------------------------------------------------------

class PolarimetricAttention(nn.Module):
    """Channel-wise cross-attention for dual-polarization (VV, VH, VV-VH) interactions."""

    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class Sentinel1SARNet(nn.Module):
    """Deep polarimetric neural network fine-tuned on Sentinel-1 SAR imagery."""

    def __init__(
        self,
        num_classes: int = 7,
        vocab_size: int = 64,
        embed_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        # 1. Multi-scale Convolutional Backbone
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 2. Polarimetric Attention Mechanism
        self.pol_attn = PolarimetricAttention(256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 3. Visual Projection
        self.visual_proj = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 4. Land-Cover Classification Head (CORINE Categories)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

        # 5. Question Tokenizer & VQA Reasoning Head
        self.text_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.text_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.vqa_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, vocab_size),
        )

        # 6. Physical Parameter Regressor (predicts mean VV dB, VH dB, and double-bounce index)
        self.regress_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),  # [vv_db, vh_db, double_bounce_index]
        )

    def forward(
        self,
        images: torch.Tensor,
        question_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = self.conv1(images)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pol_attn(x)
        feats = self.global_pool(x).flatten(1)
        v_emb = self.visual_proj(feats)

        # 1. Classification logits
        cls_logits = self.cls_head(v_emb)

        # 2. Physical dielectric parameters
        params = self.regress_head(v_emb)

        # 3. Optional VQA reasoning
        vqa_logits = None
        if question_ids is not None:
            t_emb = self.text_embedding(question_ids)
            _, h_n = self.text_gru(t_emb)
            t_feat = h_n.squeeze(0)
            fused = torch.cat([v_emb, t_feat], dim=-1)
            vqa_logits = self.vqa_fusion(fused)

        return {
            "cls_logits": cls_logits,
            "params": params,
            "vqa_logits": vqa_logits,
            "visual_embedding": v_emb,
        }


# ---------------------------------------------------------------------------
# Dataset & Preprocessing
# ---------------------------------------------------------------------------

class Sentinel1SARDataset(Dataset):
    """PyTorch Dataset for BigEarthNet Sentinel-1 SAR tiles and QA pairs."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        data_dir: Path,
        cat2idx: Dict[str, int],
        w2i: Dict[str, int],
        ans2idx: Dict[str, int],
        transform: Any,
    ):
        self.samples = samples
        self.data_dir = data_dir
        self.cat2idx = cat2idx
        self.w2i = w2i
        self.ans2idx = ans2idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        img_path = self.data_dir / item["image_path"]

        img = Image.open(img_path).convert("RGB")
        tensor_img = self.transform(img)

        cat_id = self.cat2idx.get(item["category"], 0)

        # Tokenize question
        q_tokens = [tok.lower() for tok in item["question"].split()[:16]]
        q_ids = [self.w2i.get(tok, 1) for tok in q_tokens]
        if len(q_ids) < 16:
            q_ids += [0] * (16 - len(q_ids))
        q_tensor = torch.tensor(q_ids, dtype=torch.long)

        # Target answer ID
        ans_id = self.ans2idx.get(item["answer"].lower(), 0)

        # Physical parameters
        meta = item.get("metadata", {})
        vv_val = float(meta.get("mean_vv_db", -15.0))
        vh_val = float(meta.get("mean_vh_db", -22.0))
        db_index = 1.0 if item["category"] in ("urban", "industrial") else 0.0
        params_target = torch.tensor([vv_val, vh_val, db_index], dtype=torch.float32)

        return {
            "image": tensor_img,
            "cat_id": torch.tensor(cat_id, dtype=torch.long),
            "question_ids": q_tensor,
            "answer_id": torch.tensor(ans_id, dtype=torch.long),
            "params": params_target,
        }


# ---------------------------------------------------------------------------
# Training & Evaluation Engine
# ---------------------------------------------------------------------------

def train_sentinel1_sar_model(
    data_dir: str = "./datasets/bigearthnet_sar",
    output_dir: str = "./models/optical_sar",
    epochs: int = 6,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "auto",
) -> Dict[str, Any]:
    """Train Sentinel1SARNet on BigEarthNet-SAR dataset."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    print(f"\n[INFO] Starting Sentinel-1 SAR Model Training on {dev}...")

    # 1. Load dataset metadata & splits
    with open(data_path / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    categories = meta["categories"]
    cat2idx = {cat: i for i, cat in enumerate(categories)}
    idx2cat = {i: cat for i, cat in enumerate(categories)}

    with open(data_path / "train.json", "r", encoding="utf-8") as f:
        train_samples = json.load(f)
    with open(data_path / "val.json", "r", encoding="utf-8") as f:
        val_samples = json.load(f)
    with open(data_path / "test.json", "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    # 2. Build answer & question vocabularies
    all_answers = sorted(list(set(s["answer"].lower() for s in train_samples + val_samples)))
    ans2idx = {ans: i for i, ans in enumerate(all_answers)}
    idx2ans = {i: ans for i, ans in enumerate(all_answers)}

    w2i = {"<pad>": 0, "<unk>": 1}
    for s in train_samples:
        for w in s["question"].lower().split():
            if w not in w2i:
                w2i[w] = len(w2i)
    idx2word = {i: w for w, i in w2i.items()}

    # 3. Transforms
    transform_train = T.Compose([
        T.Resize((128, 128)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_eval = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = Sentinel1SARDataset(train_samples, data_path, cat2idx, w2i, ans2idx, transform_train)
    val_ds = Sentinel1SARDataset(val_samples, data_path, cat2idx, w2i, ans2idx, transform_eval)
    test_ds = Sentinel1SARDataset(test_samples, data_path, cat2idx, w2i, ans2idx, transform_eval)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 4. Instantiate Model
    model = Sentinel1SARNet(
        num_classes=len(categories),
        vocab_size=max(len(w2i), len(ans2idx)) + 10,
        embed_dim=128,
        hidden_dim=256,
    ).to(dev)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_vqa = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    best_val_acc = 0.0
    best_weights = None

    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_cls = 0
        total_cls = 0

        for batch in train_loader:
            imgs = batch["image"].to(dev)
            cat_targets = batch["cat_id"].to(dev)
            q_ids = batch["question_ids"].to(dev)
            ans_targets = batch["answer_id"].to(dev)
            params_targets = batch["params"].to(dev)

            optimizer.zero_grad()
            outputs = model(imgs, question_ids=q_ids)

            loss_cls = criterion_cls(outputs["cls_logits"], cat_targets)
            loss_vqa = criterion_vqa(outputs["vqa_logits"], ans_targets)
            loss_reg = criterion_reg(outputs["params"], params_targets)

            loss = loss_cls + 0.8 * loss_vqa + 0.1 * loss_reg
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs["cls_logits"].argmax(dim=-1)
            correct_cls += (preds == cat_targets).sum().item()
            total_cls += imgs.size(0)

        train_acc = correct_cls / max(1, total_cls)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(dev)
                cat_targets = batch["cat_id"].to(dev)
                outputs = model(imgs)
                preds = outputs["cls_logits"].argmax(dim=-1)
                val_correct += (preds == cat_targets).sum().item()
                val_total += imgs.size(0)

        val_acc = val_correct / max(1, val_total)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {total_loss / total_cls:.4f} | Train Acc: {train_acc * 100:.1f}% | Val Acc: {val_acc * 100:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}

    # 5. Final Evaluation on Held-Out Test Set
    if best_weights:
        model.load_state_dict(best_weights)
    model.eval()
    model.to(dev)

    test_correct = 0
    test_total = 0
    all_preds, all_gts = [], []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(dev)
            cat_targets = batch["cat_id"].to(dev)
            outputs = model(imgs)
            preds = outputs["cls_logits"].argmax(dim=-1)
            test_correct += (preds == cat_targets).sum().item()
            test_total += imgs.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_gts.extend(cat_targets.cpu().tolist())

    test_acc = test_correct / max(1, test_total)

    # Per-class metrics
    class_accuracies = {}
    for cat, idx in cat2idx.items():
        cat_mask = [g == idx for g in all_gts]
        if any(cat_mask):
            cat_correct = sum(1 for p, g in zip(all_preds, all_gts) if g == idx and p == idx)
            class_accuracies[cat] = round(cat_correct / sum(cat_mask), 4)

    # Calculate Macro F1
    precisions, recalls = [], []
    for idx in range(len(categories)):
        tp = sum(1 for p, g in zip(all_preds, all_gts) if p == idx and g == idx)
        fp = sum(1 for p, g in zip(all_preds, all_gts) if p == idx and g != idx)
        fn = sum(1 for p, g in zip(all_preds, all_gts) if p != idx and g == idx)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        precisions.append(prec)
        recalls.append(rec)

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = (
        2.0 * macro_precision * macro_recall / max(1e-5, macro_precision + macro_recall)
    )

    elapsed = time.perf_counter() - t_start

    print(f"\n[EVALUATION REPORT — BigEarthNet Sentinel-1 SAR Model]")
    print(f"Test Overall Accuracy (OA): {test_acc * 100:.2f}%")
    print(f"Test Macro F1:              {macro_f1:.4f}")
    print(f"Test Macro Precision:       {macro_precision:.4f}")
    print(f"Test Macro Recall:          {macro_recall:.4f}")
    print(f"Per-Class Accuracies:       {class_accuracies}")

    # 6. Save Model Checkpoint & Metadata
    model_checkpoint_path = out_path / "sar_sentinel1_adapter.pt"
    torch.save(
        {
            "model_state_dict": best_weights or model.state_dict(),
            "config": {
                "num_classes": len(categories),
                "vocab_size": max(len(w2i), len(ans2idx)) + 10,
                "embed_dim": 128,
                "hidden_dim": 256,
            },
            "cat2idx": cat2idx,
            "idx2cat": idx2cat,
            "ans2idx": ans2idx,
            "idx2ans": idx2ans,
            "w2i": w2i,
            "idx2word": idx2word,
            "metrics": {
                "test_accuracy": round(test_acc, 4),
                "macro_f1": round(macro_f1, 4),
                "macro_precision": round(macro_precision, 4),
                "macro_recall": round(macro_recall, 4),
                "class_accuracies": class_accuracies,
            },
        },
        model_checkpoint_path,
    )

    info_meta = {
        "model_name": "sar-sentinel1-specialist",
        "version": "1.0.0",
        "dataset": "BigEarthNet-Sentinel-1-SAR",
        "architecture": "Sentinel1SARNet (Dual-Pol Attentive ResNet + Polarimetric Head)",
        "training_epochs": epochs,
        "batch_size": batch_size,
        "device": str(dev),
        "training_duration_seconds": round(elapsed, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_metrics": {
            "overall_accuracy": round(test_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "per_class_accuracy": class_accuracies,
        },
    }

    with open(out_path / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(info_meta, f, indent=2)

    print(f"[SUCCESS] Saved trained Sentinel-1 SAR model to {model_checkpoint_path}")
    return info_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dedicated Sentinel-1 SAR polarimetric model")
    parser.add_argument("--data-dir", type=str, default="./datasets/bigearthnet_sar")
    parser.add_argument("--output-dir", type=str, default="./models/optical_sar")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    train_sentinel1_sar_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

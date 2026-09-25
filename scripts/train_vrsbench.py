"""SatQuery AI — VRSBench Specialist Training Pipeline.

Trains parameter-efficient remote-sensing adapters on VRSBench:
1. Visual Grounding: Text-conditioned bounding-box regression [ymin, xmin, ymax, xmax]
   trained on referring expressions from VRSBench_EVAL_referring.json.
   Outputs -> models/grounding/vrsbench_grounding_adapter.pt

2. Scene Captioning: Visual feature conditioned sequence generator
   trained on scene captions from VRSBench_EVAL_Cap.json.
   Outputs -> models/captioning/vrsbench_caption_adapter.pt
"""

import argparse
import io
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import zipfile

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Text Vocabularies
# ---------------------------------------------------------------------------

def simple_tokenize(text: str, max_len: int = 32) -> List[str]:
    tokens = re.findall(r"\w+", text.lower())
    return tokens[:max_len]


class TextVocab:
    def __init__(self, pad_token="<pad>", unk_token="<unk>", start_token="<bos>", end_token="<eos>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.start_token = start_token
        self.end_token = end_token
        self.w2i = {pad_token: 0, unk_token: 1, start_token: 2, end_token: 3}
        self.i2w = {0: pad_token, 1: unk_token, 2: start_token, 3: end_token}

    def fit(self, texts: List[str], max_vocab: int = 2000):
        from collections import Counter
        counter = Counter()
        for t in texts:
            counter.update(simple_tokenize(t, max_len=64))
        for word, _ in counter.most_common(max_vocab):
            if word not in self.w2i:
                idx = len(self.w2i)
                self.w2i[word] = idx
                self.i2w[idx] = word

    def encode(self, text: str, max_len: int = 32, add_special_tokens: bool = False) -> List[int]:
        tokens = simple_tokenize(text, max_len)
        ids = []
        if add_special_tokens:
            ids.append(self.w2i[self.start_token])
        ids.extend([self.w2i.get(tok, self.w2i[self.unk_token]) for tok in tokens])
        if add_special_tokens:
            ids.append(self.w2i[self.end_token])
        if len(ids) < max_len:
            ids = ids + [self.w2i[self.pad_token]] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids: List[int]) -> str:
        words = []
        for i in ids:
            if i in (self.w2i[self.pad_token], self.w2i[self.start_token]):
                continue
            if i == self.w2i[self.end_token]:
                break
            words.append(self.i2w.get(i, self.unk_token))
        return " ".join(words)


# ---------------------------------------------------------------------------
# Visual Grounding Model & Training
# ---------------------------------------------------------------------------

class VRSGroundingModel(nn.Module):
    """Text-guided bounding box regression model for remote-sensing grounding."""
    def __init__(self, vocab_size: int, embed_dim: int = 64):
        super().__init__()
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, 128, batch_first=True)

        self.box_head = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid(),  # Box coordinates normalized [0, 1]
        )

    def forward(self, img: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        v_feat = self.visual_encoder(img)
        emb = self.embedding(text_ids)
        _, h_n = self.gru(emb)
        t_feat = h_n.squeeze(0)
        fused = torch.cat([v_feat, t_feat], dim=1)
        return self.box_head(fused)  # (B, 4) -> [ymin, xmin, ymax, xmax]


def parse_grounding_box(gt_str: str) -> Optional[List[float]]:
    """Parse box string like '{<25><40><33><60>}' (xmin, ymin, xmax, ymax) into [ymin, xmin, ymax, xmax]."""
    nums = re.findall(r"\d+", gt_str)
    if len(nums) == 4:
        x1, y1, x2, y2 = [float(n) / 100.0 for n in nums]
        ymin = min(y1, y2)
        xmin = min(x1, x2)
        ymax = max(y1, y2)
        xmax = max(x1, x2)
        return [ymin, xmin, ymax, xmax]
    return None


class VRSGroundingDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], zip_path: Path, vocab: TextVocab, transform: T.Compose):
        self.samples = samples
        self.zip_path = str(zip_path)
        self.vocab = vocab
        self.transform = transform
        self._zf = None

    def _get_zf(self):
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path, "r")
        return self._zf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        zf = self._get_zf()
        with zf.open(f"Images_val/{item['image_id']}") as f:
            with Image.open(io.BytesIO(f.read())) as pil_img:
                img_t = self.transform(pil_img.convert("RGB"))

        text_ids = torch.tensor(self.vocab.encode(item["question"], max_len=24), dtype=torch.long)
        box = torch.tensor(item["box"], dtype=torch.float32)
        return img_t, text_ids, box


def train_grounding(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 32,
    limit_samples: int = 1500,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
):
    print("\n" + "=" * 60)
    print("Training VRSBench Visual Grounding Specialist")
    print("=" * 60)

    ref_file = data_dir / "VRSBench_EVAL_referring.json"
    zip_path = data_dir / "Images_val.zip"

    if not (ref_file.exists() and zip_path.exists()):
        print(f"[ERROR] Missing VRSBench files in {data_dir}")
        return

    with open(ref_file, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    # Filter and parse valid boxes
    samples = []
    with zipfile.ZipFile(zip_path) as zf:
        valid_images = set(n.split("/")[-1] for n in zf.namelist() if n.endswith(".png"))

    for r in raw_samples:
        if r.get("image_id") not in valid_images:
            continue
        box = parse_grounding_box(r.get("ground_truth", ""))
        if box is not None:
            samples.append({
                "image_id": r["image_id"],
                "question": r["question"],
                "box": box,
            })
        if limit_samples and len(samples) >= limit_samples:
            break

    print(f"Loaded {len(samples)} visual grounding samples from VRSBench.")
    if len(samples) == 0:
        return

    split_idx = int(len(samples) * 0.85)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    vocab = TextVocab()
    vocab.fit([s["question"] for s in train_samples])

    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = VRSGroundingDataset(train_samples, zip_path, vocab, transform)
    val_ds = VRSGroundingDataset(val_samples, zip_path, vocab, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = VRSGroundingModel(vocab_size=len(vocab.w2i), embed_dim=64).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Grounding Epoch {epoch}/{epochs}", ncols=80)
        for imgs, t_ids, boxes in pbar:
            imgs, t_ids, boxes = imgs.to(device), t_ids.to(device), boxes.to(device)
            optimizer.zero_grad()
            preds = model(imgs, t_ids)
            loss = criterion(preds, boxes)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation IoU
        model.eval()
        ious = []
        with torch.no_grad():
            for imgs, t_ids, boxes in val_loader:
                imgs, t_ids, boxes = imgs.to(device), t_ids.to(device), boxes.to(device)
                preds = model(imgs, t_ids)
                p_np = preds.cpu().numpy()
                b_np = boxes.cpu().numpy()
                for p, b in zip(p_np, b_np):
                    # compute IoU
                    yA = max(p[0], b[0])
                    xA = max(p[1], b[1])
                    yB = min(p[2], b[2])
                    xB = min(p[3], b[3])
                    inter = max(0.0, yB - yA) * max(0.0, xB - xA)
                    areaA = max(1e-4, (p[2] - p[0]) * (p[3] - p[1]))
                    areaB = max(1e-4, (b[2] - b[0]) * (b[3] - b[1]))
                    iou = inter / float(areaA + areaB - inter + 1e-6)
                    ious.append(iou)

        mean_val_iou = float(np.mean(ious)) if ious else 0.0
        print(f"Epoch {epoch}: Train Loss = {total_loss / len(train_ds):.4f}, Val Mean IoU = {mean_val_iou:.4f}")

    # Save Grounding Checkpoint
    out_dir = Path("models/grounding")
    out_dir.mkdir(parents=True, exist_ok=True)
    chk_path = out_dir / "vrsbench_grounding_adapter.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "word2idx": vocab.w2i,
        "idx2word": vocab.i2w,
        "config": {"vocab_size": len(vocab.w2i), "embed_dim": 64},
        "mean_val_iou": round(mean_val_iou, 4),
    }, chk_path)
    print(f"[SUCCESS] Saved Grounding Checkpoint -> {chk_path}")


# ---------------------------------------------------------------------------
# Scene Captioning Model & Training
# ---------------------------------------------------------------------------

class VRSCaptioningModel(nn.Module):
    """Visual feature-conditioned sequence decoder for scene captioning."""
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.decoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, img: torch.Tensor, caption_ids: torch.Tensor) -> torch.Tensor:
        v_feat = self.visual_encoder(img)  # (B, hidden_dim)
        h_0 = v_feat.unsqueeze(0)          # Initial hidden state (1, B, hidden_dim)
        emb = self.embedding(caption_ids)  # (B, L, embed_dim)
        out, _ = self.decoder_gru(emb, h_0)
        logits = self.fc_out(out)          # (B, L, vocab_size)
        return logits

    def generate_caption(self, img: torch.Tensor, vocab: TextVocab, max_len: int = 30) -> str:
        with torch.no_grad():
            v_feat = self.visual_encoder(img)
            h = v_feat.unsqueeze(0)
            curr_id = torch.tensor([[vocab.w2i[vocab.start_token]]], dtype=torch.long, device=img.device)
            gen_ids = []
            for _ in range(max_len):
                emb = self.embedding(curr_id)
                out, h = self.decoder_gru(emb, h)
                logits = self.fc_out(out)
                next_id = logits.argmax(dim=-1).item()
                if next_id == vocab.w2i[vocab.end_token]:
                    break
                gen_ids.append(next_id)
                curr_id = torch.tensor([[next_id]], dtype=torch.long, device=img.device)
            return vocab.decode(gen_ids)


class VRSCaptionDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], zip_path: Path, vocab: TextVocab, transform: T.Compose):
        self.samples = samples
        self.zip_path = str(zip_path)
        self.vocab = vocab
        self.transform = transform
        self._zf = None

    def _get_zf(self):
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path, "r")
        return self._zf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        zf = self._get_zf()
        with zf.open(f"Images_val/{item['image_id']}") as f:
            with Image.open(io.BytesIO(f.read())) as pil_img:
                img_t = self.transform(pil_img.convert("RGB"))

        seq = self.vocab.encode(item["caption"], max_len=30, add_special_tokens=True)
        inp = torch.tensor(seq[:-1], dtype=torch.long)
        tgt = torch.tensor(seq[1:], dtype=torch.long)
        return img_t, inp, tgt


def train_captioning(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 32,
    limit_samples: int = 1500,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
):
    print("\n" + "=" * 60)
    print("Training VRSBench Scene Captioning Specialist")
    print("=" * 60)

    cap_file = data_dir / "VRSBench_EVAL_Cap.json"
    zip_path = data_dir / "Images_val.zip"

    if not (cap_file.exists() and zip_path.exists()):
        print(f"[ERROR] Missing VRSBench files in {data_dir}")
        return

    with open(cap_file, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    with zipfile.ZipFile(zip_path) as zf:
        valid_images = set(n.split("/")[-1] for n in zf.namelist() if n.endswith(".png"))

    samples = []
    for r in raw_samples:
        if r.get("image_id") in valid_images and r.get("ground_truth"):
            samples.append({
                "image_id": r["image_id"],
                "caption": r["ground_truth"],
            })
        if limit_samples and len(samples) >= limit_samples:
            break

    print(f"Loaded {len(samples)} captioning samples from VRSBench.")
    if len(samples) == 0:
        return

    split_idx = int(len(samples) * 0.85)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    vocab = TextVocab()
    vocab.fit([s["caption"] for s in train_samples], max_vocab=2500)

    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = VRSCaptionDataset(train_samples, zip_path, vocab, transform)
    val_ds = VRSCaptionDataset(val_samples, zip_path, vocab, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = VRSCaptioningModel(vocab_size=len(vocab.w2i), embed_dim=64, hidden_dim=128).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.w2i[vocab.pad_token])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Caption Epoch {epoch}/{epochs}", ncols=80)
        for imgs, inps, tgts in pbar:
            imgs, inps, tgts = imgs.to(device), inps.to(device), tgts.to(device)
            optimizer.zero_grad()
            logits = model(imgs, inps)
            loss = criterion(logits.reshape(-1, len(vocab.w2i)), tgts.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, inps, tgts in val_loader:
                imgs, inps, tgts = imgs.to(device), inps.to(device), tgts.to(device)
                logits = model(imgs, inps)
                loss = criterion(logits.reshape(-1, len(vocab.w2i)), tgts.reshape(-1))
                val_loss += loss.item() * imgs.size(0)

        print(f"Epoch {epoch}: Train Loss = {total_loss / len(train_ds):.4f}, Val Loss = {val_loss / len(val_ds):.4f}")

    # Save Captioning Checkpoint
    out_dir = Path("models/captioning")
    out_dir.mkdir(parents=True, exist_ok=True)
    chk_path = out_dir / "vrsbench_caption_adapter.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "word2idx": vocab.w2i,
        "idx2word": vocab.i2w,
        "config": {"vocab_size": len(vocab.w2i), "embed_dim": 64, "hidden_dim": 128},
    }, chk_path)
    print(f"[SUCCESS] Saved Captioning Checkpoint -> {chk_path}")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SatQuery AI — VRSBench Training Pipeline")
    parser.add_argument("--task", choices=["all", "grounding", "captioning"], default="all")
    parser.add_argument("--data-dir", type=str, default="./datasets/vrsbench")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-samples", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    if args.task in ("all", "grounding"):
        train_grounding(
            data_dir=data_dir,
            output_dir=Path("models/grounding"),
            epochs=args.epochs,
            batch_size=args.batch_size,
            limit_samples=args.limit_samples,
            lr=args.lr,
            device=device,
        )

    if args.task in ("all", "captioning"):
        train_captioning(
            data_dir=data_dir,
            output_dir=Path("models/captioning"),
            epochs=args.epochs,
            batch_size=args.batch_size,
            limit_samples=args.limit_samples,
            lr=args.lr,
            device=device,
        )


if __name__ == "__main__":
    main()

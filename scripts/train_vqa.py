"""SatQuery AI — RSVQA Model Training Pipeline.

Trains a parameter-efficient remote-sensing visual question answering adapter
on the RSVQA dataset (LR or HR).

Outputs:
- models/vqa/rsvqa_adapter.pt (Model weights & vocabulary)
- models/vqa/model_info.json (Model metadata & training statistics)
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def build_vocab(answers: List[str], max_vocab: int = 500) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build answer vocabulary from training set."""
    from collections import Counter
    counts = Counter(ans.strip().lower() for ans in answers if ans.strip())
    top_answers = [ans for ans, _ in counts.most_common(max_vocab)]
    ans2idx = {ans: idx for idx, ans in enumerate(top_answers)}
    idx2ans = {idx: ans for ans, idx in ans2idx.items()}
    return ans2idx, idx2ans


def simple_tokenize(text: str, max_len: int = 24) -> List[str]:
    """Tokenize and pad/truncate query text."""
    import re
    tokens = re.findall(r"\w+", text.lower())
    return tokens[:max_len]


class WordVocab:
    def __init__(self, pad_token="<pad>", unk_token="<unk>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.w2i = {pad_token: 0, unk_token: 1}
        self.i2w = {0: pad_token, 1: unk_token}

    def fit(self, texts: List[str]):
        for t in texts:
            for tok in simple_tokenize(t):
                if tok not in self.w2i:
                    idx = len(self.w2i)
                    self.w2i[tok] = idx
                    self.i2w[idx] = tok

    def transform(self, text: str, max_len: int = 24) -> List[int]:
        tokens = simple_tokenize(text, max_len)
        ids = [self.w2i.get(tok, self.w2i[self.unk_token]) for tok in tokens]
        if len(ids) < max_len:
            ids = ids + [self.w2i[self.pad_token]] * (max_len - len(ids))
        return ids[:max_len]


def load_dataset_samples(
    data_dir: Path,
    split: str = "train",
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Path]:
    """Load question-answer records and images folder."""
    images_dir = data_dir / "images"
    splits_dir = data_dir / "splits"
    q_file = splits_dir / f"{split}_questions.json"
    a_file = splits_dir / f"{split}_answers.json"

    if not q_file.exists():
        q_file = data_dir / "questions" / "questions.json"
    if not a_file.exists():
        a_file = data_dir / "answers" / "answers.json"

    with open(q_file, "r", encoding="utf-8") as f:
        q_data = json.load(f).get("questions", [])
    with open(a_file, "r", encoding="utf-8") as f:
        a_data = json.load(f).get("answers", [])

    ans_map = {a["id"]: str(a.get("answer", "")).strip().lower() for a in a_data if a.get("active", True)}

    samples = []
    for q in q_data:
        if not q.get("active", True):
            continue
        ans_ids = q.get("answers_ids", [])
        ans = ans_map.get(ans_ids[0]) if ans_ids else None
        if not ans:
            continue

        img_id = q.get("img_id")
        img_path = None
        for ext in (".tif", ".tiff", ".png", ".jpg"):
            cand = images_dir / f"{img_id}{ext}"
            if cand.exists():
                img_path = cand
                break

        if not img_path:
            continue

        samples.append({
            "img_path": img_path,
            "question": q.get("question", ""),
            "answer": ans,
            "category": q.get("type", "general"),
        })

        if limit and len(samples) >= limit:
            break

    return samples, images_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="SatQuery AI — RSVQA Training Pipeline")
    parser.add_argument("--data-dir", type=str, default="./datasets/rsvqa/lr")
    parser.add_argument("--output-dir", type=str, default="./models/vqa")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit-samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as T

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    )
    print("=" * 65)
    print("SatQuery AI — RSVQA Model Training")
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print("=" * 65)

    data_dir = Path(args.data_dir)
    train_samples, _ = load_dataset_samples(data_dir, split="train", limit=args.limit_samples)
    val_samples, _ = load_dataset_samples(data_dir, split="val", limit=max(50, args.limit_samples // 5))

    print(f"Loaded {len(train_samples)} training samples, {len(val_samples)} validation samples")

    if not train_samples:
        print("[ERROR] No training samples found. Please ensure the dataset is downloaded.")
        sys.exit(1)

    # Build vocabularies
    ans2idx, idx2ans = build_vocab([s["answer"] for s in train_samples])
    word_vocab = WordVocab()
    word_vocab.fit([s["question"] for s in train_samples])

    print(f"Answer vocabulary size: {len(ans2idx)}")
    print(f"Word vocabulary size: {len(word_vocab.w2i)}")

    # PyTorch Dataset
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class RSVQA_TorchDataset(Dataset):
        def __init__(self, samples):
            self.samples = [s for s in samples if s["answer"] in ans2idx]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            item = self.samples[idx]
            with Image.open(item["img_path"]) as img:
                img_t = transform(img.convert("RGB"))
            q_ids = torch.tensor(word_vocab.transform(item["question"]), dtype=torch.long)
            target = torch.tensor(ans2idx[item["answer"]], dtype=torch.long)
            return img_t, q_ids, target

    train_ds = RSVQA_TorchDataset(train_samples)
    val_ds = RSVQA_TorchDataset(val_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Lightweight Model Architecture
    class RSVQAModel(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_classes):
            super().__init__()
            # CNN Visual Feature Extractor
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
            # Question Text Encoder
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.gru = nn.GRU(embed_dim, 128, batch_first=True)

            # Fusion Head
            self.classifier = nn.Sequential(
                nn.Linear(128 + 128, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes),
            )

        def forward(self, img, text_ids):
            v_feat = self.visual_encoder(img)  # (B, 128)
            emb = self.embedding(text_ids)      # (B, L, embed_dim)
            _, h_n = self.gru(emb)
            t_feat = h_n.squeeze(0)            # (B, 128)
            fused = torch.cat([v_feat, t_feat], dim=1)
            return self.classifier(fused)

    model = RSVQAModel(
        vocab_size=len(word_vocab.w2i),
        embed_dim=64,
        num_classes=len(ans2idx),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Training Loop
    print("\n--- Starting Training ---")
    training_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=85)
        for imgs, q_ids, targets in pbar:
            imgs, q_ids, targets = imgs.to(device), q_ids.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs, q_ids)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, q_ids, targets in val_loader:
                imgs, q_ids, targets = imgs.to(device), q_ids.to(device), targets.to(device)
                outputs = model(imgs, q_ids)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_acc = val_correct / max(1, val_total)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc*100:.2f}%, Val Acc = {val_acc*100:.2f}%")
        training_history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})

    # Save Model & Artifacts
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "rsvqa_adapter.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "ans2idx": ans2idx,
        "idx2ans": idx2ans,
        "word2idx": word_vocab.w2i,
        "idx2word": word_vocab.i2w,
        "config": {
            "vocab_size": len(word_vocab.w2i),
            "embed_dim": 64,
            "num_classes": len(ans2idx),
        },
        "history": training_history,
        "final_val_acc": val_acc,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"\n[Saved Model Checkpoint] -> {checkpoint_path}")

    # Metadata report
    metadata = {
        "model_name": "rsvqa-spectral-vqa",
        "task": "single-image VQA",
        "dataset": str(data_dir.name),
        "total_training_samples": len(train_ds),
        "num_classes": len(ans2idx),
        "top_answers": list(ans2idx.keys())[:15],
        "final_validation_accuracy": round(val_acc, 4),
        "checkpoint": str(checkpoint_path.resolve()),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(output_dir / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Saved Metadata] -> {output_dir / 'model_info.json'}")
    print("=" * 65)


if __name__ == "__main__":
    main()

"""SatQuery AI — ResSingleRSVQAModel Architecture.

High-accuracy Residual Network with Cross-Modal Multimodal Gated Attention
for single-image Remote Sensing Visual Question Answering (RSVQA & VRSBench).
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual convolutional block with skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + res)


class ResSingleRSVQAModel(nn.Module):
    """High-accuracy Residual Single-Image VQA Network with cross-modal gating."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_classes: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # 1. Visual Backbone: 4-stage Residual CNN
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer1 = ResBlock(32, 64, stride=2)    # 32x32
        self.layer2 = ResBlock(64, 128, stride=2)   # 16x16
        self.layer3 = ResBlock(128, 128, stride=2)  # 8x8

        # Dual Pooling (Avg + Max for structural & spectral features)
        self.visual_proj = nn.Sequential(
            nn.Linear(128 * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # 2. Text Encoder: Bidirectional GRU
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # 3. Cross-Modal Multimodal Gating
        self.multimodal_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # 4. Multi-Modal Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def extract_visual(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        avg_pool = F.adaptive_avg_pool2d(h, (1, 1)).flatten(1)
        max_pool = F.adaptive_max_pool2d(h, (1, 1)).flatten(1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        return self.visual_proj(pooled)

    def extract_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(text_ids)
        _, h_n = self.gru(emb)
        # h_n shape: (2, B, hidden_dim // 2) -> concat directions
        text_feat = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.text_proj(text_feat)

    def forward(self, img: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        v_feat = self.extract_visual(img)      # (B, hidden_dim)
        t_feat = self.extract_text(text_ids)   # (B, hidden_dim)

        gate = self.multimodal_gate(torch.cat([v_feat, t_feat], dim=-1))
        gated_v = v_feat * gate
        interaction = v_feat * t_feat
        fused = torch.cat([gated_v, t_feat, interaction], dim=-1)

        return self.classifier(fused)

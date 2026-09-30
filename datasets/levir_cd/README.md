# LEVIR-CD: Large-scale Building Change Detection Dataset

The **LEVIR-CD** dataset is a benchmark Earth Observation dataset specifically curated for evaluating and training bi-temporal building change detection algorithms on very-high-resolution (VHR) remote sensing imagery.

- **Google Drive Distribution:** [LEVIR-CD Folder](https://drive.google.com/drive/folders/1dLuzldMRmbBNKPpUkX8Z53hi6NHLrWim)
- **Reference:** Chen, H., & Shi, Z. (2020). *A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection*. Remote Sensing, 12(10), 1662.

## Key Dataset Characteristics

| Attribute | Value |
| :--- | :--- |
| **Spatial Resolution (GSD)** | **0.5 meters / pixel** |
| **Patch Size** | **$1024 \times 1024$ pixels** |
| **Total Image Pairs** | **637 pairs** ($T_1$ and $T_2$) |
| **Temporal Span** | 5 to 14 years difference between captures |
| **Ground Truth Class** | Binary Building Change Mask (`0`: unchanged, `255`: building change) |
| **Instances** | Over 31,333 individual annotated building change structures |
| **Default Split** | Train: 445 pairs, Val: 64 pairs, Test: 128 pairs |

## Directory Structure

```text
datasets/levir_cd/
├── metadata.json
├── README.md
├── train/
│   ├── A/       # Pre-event optical imagery (T1)
│   ├── B/       # Post-event optical imagery (T2)
│   └── label/   # Binary building change ground truth masks
├── val/
│   ├── A/
│   ├── B/
│   └── label/
└── test/
    ├── A/
    ├── B/
    └── label/
```

## SatQuery AI Integration

SatQuery AI natively integrates LEVIR-CD for:
1. **Model Evaluation:** Computing Change IoU, Precision, Recall, F1-score, Overall Accuracy, and Cohen's Kappa via `app.evaluation.levir_cd_evaluator.LEVIRCDCEvaluator`.
2. **Methodology Verification:** Validating the **HSPD-Change** (Hierarchical Structural-Phenological Decoupled Change Engine) and **BIT** (Bitemporal Image Transformer) models.
3. **Automated Benchmark CLI:**
   ```bash
   python -m app.evaluation.runner --benchmark levir_cd
   ```
4. **Data Ingestion Script:**
   ```bash
   python scripts/download_levir_cd.py --download-drive
   ```

# SatQuery AI — Dataset Configuration

## 1. Overview

SatQuery AI integrates four benchmark remote-sensing datasets for model training, adaptation, and evaluation. Each dataset has a distinct format and requires a dedicated loader.

---

## 2. Dataset Summary

| Dataset | Purpose | Tasks | Format | Size (approx.) |
|---------|---------|-------|--------|----------------|
| BigEarthNet | RS image-text adaptation | Multi-label classification | GeoTIFF + labels | ~66GB (S2) |
| VRSBench | Captioning, Grounding, VQA | Caption, Ground, VQA | Images + JSON | ~5GB |
| RSVQA | Single-image VQA | VQA | Images + QA pairs | ~1-15GB |
| CDVQA | Change-based VQA | Change VQA | Image pairs + QA | ~3GB |

---

## 3. BigEarthNet

### 3.1 Description

BigEarthNet is a large-scale benchmark for multi-label remote-sensing image classification. In SatQuery AI, we use BigEarthNet with text descriptions (BigEarthNet-MM / BigEarthNet-S2) for vision-language model adaptation.

### 3.2 Source

- **URL:** https://bigearth.net/
- **Paper:** Sumbul et al., "BigEarthNet: A Large-Scale Benchmark Archive for Remote Sensing Image Understanding" (2019)

### 3.3 Format

```
bigearthnet_txt/
├── S2A_MSIL2A_20170613T101031_0_45/
│   ├── S2A_MSIL2A_20170613T101031_0_45_B01.tif
│   ├── S2A_MSIL2A_20170613T101031_0_45_B02.tif
│   ├── ...
│   ├── S2A_MSIL2A_20170613T101031_0_45_B12.tif
│   └── S2A_MSIL2A_20170613T101031_0_45_labels_metadata.json
├── ...
└── metadata.csv
```

### 3.4 Bands (Sentinel-2)

| Band | Name | Resolution | Wavelength |
|------|------|-----------|------------|
| B01 | Coastal aerosol | 60m | 443nm |
| B02 | Blue | 10m | 490nm |
| B03 | Green | 10m | 560nm |
| B04 | Red | 10m | 665nm |
| B05 | Veg. Red Edge 1 | 20m | 705nm |
| B06 | Veg. Red Edge 2 | 20m | 740nm |
| B07 | Veg. Red Edge 3 | 20m | 783nm |
| B08 | NIR | 10m | 842nm |
| B08A | Narrow NIR | 20m | 865nm |
| B09 | Water Vapour | 60m | 945nm |
| B11 | SWIR 1 | 20m | 1610nm |
| B12 | SWIR 2 | 20m | 2190nm |

### 3.5 Labels

19 classes (CORINE Land Cover level-3):
- Continuous urban fabric, Discontinuous urban fabric, Industrial/commercial, Road/rail, Port, Airport, Mineral extraction, Dump, Construction, Green urban, Sport/leisure, Non-irrigated arable, Permanently irrigated, Rice fields, Vineyards, Fruit trees, Pastures, Complex cultivation, Agriculture with natural vegetation

### 3.6 Loader Configuration

```yaml
bigearthnet:
  root_dir: "./datasets/bigearthnet_txt/"
  bands: [B02, B03, B04, B08]  # RGB + NIR for default processing
  all_bands: [B01, B02, B03, B04, B05, B06, B07, B08, B08A, B09, B11, B12]
  image_size: 120  # 120x120 pixels at 10m
  num_classes: 19
  split_file: "./datasets/bigearthnet_txt/splits/"
  normalization: "per_band_minmax"
  text_labels: true  # Use text descriptions for VLM adaptation
```

### 3.7 Usage in SatQuery AI

- **Model adaptation:** LoRA fine-tuning of BLIP-2 for remote-sensing vocabulary
- **Training:** Generate image-text pairs from BigEarthNet labels for VLM adaptation
- **Validation:** Evaluate adapted model on held-out BigEarthNet samples

---

## 4. VRSBench

### 4.1 Description

VRSBench (Visual Remote Sensing Benchmark) provides annotations for captioning, visual grounding, and VQA tasks on remote-sensing imagery.

### 4.2 Source

- **Paper:** Li et al., "VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding" (2024)
- **URL:** https://github.com/lx709/VRSBench

### 4.3 Format

```
vrsbench/
├── images/
│   ├── image_0001.png
│   ├── image_0002.png
│   └── ...
├── annotations/
│   ├── captions.json
│   ├── grounding.json
│   └── vqa.json
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### 4.4 Annotation Formats

**Captions (`captions.json`):**
```json
[
  {
    "image_id": "image_0001",
    "caption": "An aerial view of a residential area with dense housing..."
  }
]
```

**Grounding (`grounding.json`):**
```json
[
  {
    "image_id": "image_0001",
    "phrase": "swimming pool",
    "bbox": [x1, y1, x2, y2],
    "category": "water_feature"
  }
]
```

**VQA (`vqa.json`):**
```json
[
  {
    "image_id": "image_0001",
    "question": "How many buildings are visible?",
    "answer": "approximately 15",
    "question_type": "counting"
  }
]
```

### 4.5 Loader Configuration

```yaml
vrsbench:
  root_dir: "./datasets/vrsbench/"
  images_dir: "./datasets/vrsbench/images/"
  annotations_dir: "./datasets/vrsbench/annotations/"
  splits_dir: "./datasets/vrsbench/splits/"
  tasks:
    - captioning
    - grounding
    - vqa
  image_size: 512
  normalization: "imagenet"
```

---

## 5. RSVQA

### 5.1 Description

RSVQA (Remote Sensing Visual Question Answering) provides question-answer pairs for single remote-sensing images, testing spatial reasoning and land-cover understanding.

### 5.2 Source

- **Paper:** Lobry et al., "RSVQA: Visual Question Answering for Remote Sensing Data" (2020)
- **URL:** https://rsvqa.sylvainlobry.com/

### 5.3 Variants

| Variant | Resolution | Images | QA Pairs |
|---------|-----------|--------|----------|
| RSVQA-LR | Low Resolution (Sentinel-2) | ~772 | ~77,232 |
| RSVQA-HR | High Resolution (aerial) | ~10,659 | ~1,066,316 |

### 5.4 Format

```
rsvqa/
├── lr/
│   ├── images/
│   │   ├── 0.tif
│   │   ├── 1.tif
│   │   └── ...
│   ├── questions/
│   │   └── questions.json
│   ├── answers/
│   │   └── answers.json
│   └── splits/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── hr/
│   └── ... (same structure)
```

### 5.5 Question Types

- **Presence:** "Is there a river in the image?"
- **Comparison:** "Is the urban area larger than the forest?"
- **Counting:** "How many buildings are visible?"
- **Area:** "What percentage of the image is water?"
- **Rural/Urban:** "Is this a rural or urban area?"

### 5.6 Loader Configuration

```yaml
rsvqa:
  root_dir: "./datasets/rsvqa/"
  variants:
    lr:
      images_dir: "./datasets/rsvqa/lr/images/"
      questions_file: "./datasets/rsvqa/lr/questions/questions.json"
      answers_file: "./datasets/rsvqa/lr/answers/answers.json"
      splits_dir: "./datasets/rsvqa/lr/splits/"
    hr:
      images_dir: "./datasets/rsvqa/hr/images/"
      questions_file: "./datasets/rsvqa/hr/questions/questions.json"
      answers_file: "./datasets/rsvqa/hr/answers/answers.json"
      splits_dir: "./datasets/rsvqa/hr/splits/"
  default_variant: "lr"
  normalization: "per_band_minmax"
```

---

## 6. CDVQA

### 6.1 Description

CDVQA (Change Detection Visual Question Answering) provides question-answer pairs about changes between bi-temporal remote-sensing image pairs.

### 6.2 Source

- **Paper:** Yuan et al., "Change Detection Meets Visual Question Answering" (2022)
- **URL:** https://github.com/YZHJessica/CDVQA

### 6.3 Format

```
cdvqa/
├── images/
│   ├── A/                    # Time 1 images
│   │   ├── 00001.png
│   │   └── ...
│   └── B/                    # Time 2 images
│       ├── 00001.png
│       └── ...
├── annotations/
│   ├── qa_pairs.json
│   └── change_maps/         # Ground truth change masks
│       ├── 00001.png
│       └── ...
└── splits/
    ├── train.json
    ├── val.json
    └── test.json
```

### 6.4 QA Format

```json
[
  {
    "image_id": "00001",
    "image_A": "A/00001.png",
    "image_B": "B/00001.png",
    "question": "Has the number of buildings increased?",
    "answer": "yes",
    "question_type": "change_presence",
    "change_type": "building_increase"
  }
]
```

### 6.5 Question Types

- **Change Presence:** "Did any change occur?"
- **Change Type:** "What type of change occurred?"
- **Change Location:** "Where did the change occur?"
- **Change Quantity:** "How much area changed?"
- **Change Direction:** "Has the built-up area increased or decreased?"

### 6.6 Loader Configuration

```yaml
cdvqa:
  root_dir: "./datasets/cdvqa/"
  images_A_dir: "./datasets/cdvqa/images/A/"
  images_B_dir: "./datasets/cdvqa/images/B/"
  annotations_file: "./datasets/cdvqa/annotations/qa_pairs.json"
  change_maps_dir: "./datasets/cdvqa/annotations/change_maps/"
  splits_dir: "./datasets/cdvqa/splits/"
  image_size: 256
  normalization: "imagenet"
```

---

## 7. Dataset Loader Interface

All dataset loaders implement a common interface:

```python
class DatasetLoader(ABC):
    """Base interface for all dataset loaders."""

    @abstractmethod
    def load_split(self, split: str) -> Dataset:
        """Load a specific split (train/val/test)."""
        ...

    @abstractmethod
    def get_sample(self, index: int) -> Dict:
        """Get a single sample by index."""
        ...

    @abstractmethod
    def get_statistics(self) -> Dict:
        """Get dataset statistics (size, class distribution, etc.)."""
        ...

    @abstractmethod
    def validate(self) -> bool:
        """Validate that the dataset exists and is properly formatted."""
        ...
```

---

## 8. Data Download Scripts

```bash
# Download scripts (to be created in scripts/)
python scripts/download_bigearthnet.py --output ./datasets/bigearthnet_txt/
python scripts/download_vrsbench.py --output ./datasets/vrsbench/
python scripts/download_rsvqa.py --variant lr --output ./datasets/rsvqa/
python scripts/download_cdvqa.py --output ./datasets/cdvqa/
```

---

## 9. Important Notes

- Large datasets should **never** be committed to Git
- Download scripts handle extraction and directory organization
- Each loader validates the dataset on initialization and reports missing files
- Normalization parameters are dataset-specific and pre-computed
- Splits (train/val/test) follow the original dataset authors' recommendations

# SatQuery AI — IEEE-Style Presentation

This directory contains the complete LaTeX Beamer presentation for **SatQuery AI** styled according to official **IEEE color themes and symposium standards** (`#006699`, `#002855`).

## Files
- `satquery_ieee_presentation.tex`: Full 17-slide presentation deck (16:9 widescreen, TikZ flowcharts, mathematical equations, benchmark comparison tables, and visual evidence workflows).

---

## Key Features Added

### 1. Dedicated Multi-Slide Literature Review
- **Slide 1: Evolution of Remote Sensing VLMs**:
  - *Phase 1: Contrastive Pretraining (2022–2023)*: RemoteCLIP, SkyCLIP (RGB alignment, zero conversational capability).
  - *Phase 2: Instruction-Tuned RS-LLMs (2023–2024)*: GeoChat, EarthGPT, SkySense (monolithic backbones blind to microwave radar physics).
  - *Phase 3: Visual Grounding & VQA Benchmarks*: RSVQA, VRSBench, CDVQA.
- **Slide 2: Modality-Specific Foundations & Paradigms**:
  - SAR Radar Polarimetry & Backscatter (Cloude-Pottier, BigEarthNet-MM Sentinel-1).
  - Multispectral Radiometry (Sentinel-2 $NDVI, NDWI$, CIR composites).
  - Bi-Temporal Change Detection (BIT-CD, ChangeFormer, seasonal phenological confounding).
  - Cross-Modal Optical-SAR Fusion (Sen1-2, all-weather cloud penetration).
- **Slide 3: Systematic Taxonomy & Research Gap Analysis**:
  - Comprehensive comparison matrix comparing Generic VLMs, RS-VLMs, Standalone Specialists, and SatQuery AI.
  - Clear taxonomy of 3 foundational gaps solved: *Modality Blindness*, *Environmental Confounding*, and *Hallucination Liability*.

### 2. Single-Page Flowcharts (Guaranteed Zero Slide Overflow)
Every flowchart is built with native TikZ and constrained using:
```latex
\adjustbox{max width=0.96\textwidth, max totalheight=0.66\textheight}{ ... }
```
This guarantees that **no flowchart exceeds the slide boundaries horizontally or vertically**:
1. **Flowchart 1: End-to-End Agentic Execution Pipeline**: Complete query ingestion $\to$ agentic planner $\to$ registry router $\to$ 4 decoupled specialists $\to$ synthesis \& hallucination guard $\to$ verified intelligence.
2. **Flowchart 2: AG-MFD Cross-Modal Cloud-Penetration Engine**: Dual-stream optical cloud segmentation (DSCI) + SAR speckle-filtered Laplacian pyramid fusion.
3. **Flowchart 3: HSPD Bi-Temporal Change Detection Pipeline**: Radiometric normalization $\to$ phenology filter $\to$ gradient structural tensor $\to$ cluster delineation.
4. **Flowchart 4: Dual-VLM Arbiter & Hallucination Guard Pipeline**: Qwen2-VL-2B + RSFlorence-2 cross-consensus arbiter anchored by physical radiometric measurements.

---

## How to Compile

### Option 1: On Overleaf (Recommended)
1. Go to [Overleaf](https://www.overleaf.com/).
2. Create a new blank project.
3. Upload `satquery_ieee_presentation.tex` (or copy-paste its content into `main.tex`).
4. Set compiler to `pdfLaTeX` (default) and click **Recompile**.
5. All required packages (`beamer`, `tikz`, `adjustbox`, `booktabs`, `tabularx`, `pgfplots`) are included in standard TeX Live on Overleaf.

### Option 2: Local CLI (TeX Live / MiKTeX / MacTeX)
Run `pdflatex` twice to generate the table of contents and frame numbering:
```bash
pdflatex satquery_ieee_presentation.tex
pdflatex satquery_ieee_presentation.tex
```

Or using `latexmk`:
```bash
latexmk -pdf satquery_ieee_presentation.tex
```

---

## Presentation Outline (17 Slides)
1. **Title Slide**: IEEE symposium format with keywords and author metadata.
2. **Presentation Outline**: Clean structured table of contents.
3. **Motivation & Challenges**: Earth observation bottlenecks and the SatQuery AI agentic paradigm.
4. **Literature Review 1**: Evolution of Remote Sensing VLMs (RemoteCLIP $\to$ GeoChat $\to$ VRSBench).
5. **Literature Review 2**: Modality-Specific Foundations (SAR polarimetry, optical indices, bi-temporal change).
6. **Literature Review 3**: Systematic Taxonomy & Research Gap Matrix.
7. **System Architecture Overview**: Multi-specialist orchestration principles.
8. **Flowchart 1**: End-to-End Agentic Execution Pipeline (*single-page scaled*).
9. **Flowchart 2**: AG-MFD Cross-Modal Cloud-Penetration Engine (*single-page scaled*).
10. **Flowchart 3**: HSPD Bi-Temporal Change Detection Pipeline (*single-page scaled*).
11. **Flowchart 4**: Dual-VLM Arbiter \& Hallucination Guard Pipeline (*single-page scaled*).
12. **Sentinel-1 SAR Specialist**: Polarimetric dual-band architecture, $\sigma^0$ regression, 100% OA on BigEarthNet-MM.
13. **Sentinel-2 Optical Specialist**: Spectral-spatial ConvNet, $NDVI, NDWI$, CIR composite.
14. **Consolidated Benchmark Results**: Cross-modality quantitative evaluation table.
15. **User Interface & Presets**: Modern React dashboard, split-slider comparison, query presets.
16. **Conclusion & Future Research**: Core contributions, edge quantization, hyperspectral expansion.
17. **Q&A / Thank You**: Contact and open-source GitHub repository details.

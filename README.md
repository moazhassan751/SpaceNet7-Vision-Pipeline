# SpaceNet7 Vision Pipeline

An end-to-end computer vision pipeline applied to the [SpaceNet-7 Multi-Temporal Urban Development](https://www.kaggle.com/datasets/amerii/spacenet-7-multitemporal-urban-development) satellite imagery dataset. The pipeline covers radiometric preprocessing, structural masking, texture-based classification, and deep learning recognition across 60 scenes (~1,400+ images).

---

## Pipeline Overview

```
Acquisition → Radiometric Pre-processing → Structural Masking → Texture Analysis → Deep Learning Recognition
     A1                  A1                      A2                   A3                  Final Project
```

---

## Repository Structure

```
SpaceNet7-Vision-Pipeline/
├── A1/                                          # Radiometric Pre-processing & Noise Mitigation
│   ├── AI303_SpaceNet7_Assignment1_cv.ipynb
│   ├── CV_E1_REPORT.pdf
│   └── output/
│       ├── figures/                             # PSF visualizations, filter comparisons
│       ├── psnr_ssim_all_images.csv             # Per-image PSNR & SSIM metrics
│       └── psnr_ssim_per_scene.csv              # Per-scene aggregated metrics
│
├── A2/                                          # Structural Representation & Masking
│   ├── AI303_SpaceNet7_Assignment2_cv.ipynb
│   ├── CV_ASSIGNMENT_2_Report.pdf
│   └── output_a2/
│       ├── chain_code/                          # Per-scene chain code CSVs
│       ├── convex_hull/                         # Per-scene convex hull point CSVs
│       ├── figures/                             # Mask overlays, morphological results
│       └── csv/                                 # Pipeline summaries
│
├── A3/                                          # Statistical Texture Analysis & Classification
│   ├── AI303_SpaceNet7_Assignment3_cv.ipynb
│   ├── CV_E3_REPORT.pdf
│   └── output_a3/
│       ├── classification/                      # CV results, feature importance, test results
│       ├── figures/                             # GLCM distributions, correlation heatmaps
│       └── csv/                                 # Feature vectors (GLCM + geometric)
│
└── CV_Final_DeepVisionPipeline/                 # End-to-End Deep Vision Pipeline
    ├── AI303_SpaceNet7_FinalProject_DeepVisionPipeline.ipynb
    ├── FinalProject_Report.pdf
    └── output_fp/
        ├── figures/                             # Training curves, confusion matrix, predictions
        └── csv/                                 # Radiometric metrics, masking summary, classification report
```

---

## Stage 1 — Radiometric Pre-processing & Noise Mitigation

**Dataset:** First 25 scenes (~585 images) from SpaceNet-7 train split

| Task | Details |
|---|---|
| Batch Processing | Deterministic sorted loading of all scene folders and GeoTIFF images |
| Bit-Depth Detection | Identified 16-bit (`uint16`) imagery; dtype-aware normalization to avoid false contouring |
| Filtering | Gaussian, Mean (5×5), and Median filters applied to noise-corrupted images |
| Quantification | PSNR and SSIM computed per image and aggregated per scene |
| Anti-Aliasing | Gaussian pre-filter applied before any downsampling to suppress checkerboard artifacts |
| PSF Documentation | 5×5 Gaussian kernel modeled as Point Spread Function with 3D surface visualization |

**Key outputs:** `psnr_ssim_all_images.csv`, `psnr_ssim_per_scene.csv`, before/after filter comparison figures

---

## Stage 2 — Structural Representation & Masking

**Dataset:** 25 representative scenes (one image per scene)

| Task | Details |
|---|---|
| Segmentation | Canny edge detection (primary) and Sobel gradient magnitude (comparison) |
| Morphological Cleaning | Erosion, Dilation, Opening, Closing with 5×5 elliptical structuring element |
| Chain Code | 8-directional Freeman chain code on largest contour boundary |
| First Difference | Rotationally invariant derivative of chain code: `D_i = (C_{i+1} - C_i) mod 8` |
| Shape Number | Minimum-magnitude rotation of first difference — starting-point invariant descriptor |
| Convex Hull | Graham Scan algorithm (O(n log n)) with cross-product turn detection |

**Key outputs:** Per-scene chain code CSVs, convex hull point CSVs, morphological comparison figures, hull overlay visualizations

---

## Stage 3 — Statistical Texture Analysis & Classification

**Dataset:** All 25 scenes × ~23 images = ~585 images (full batch)

| Task | Details |
|---|---|
| GLCM Computation | Gray-Level Co-occurrence Matrix on masked regions at offset (1, 0°) |
| Statistical Descriptors | Energy, Entropy, Contrast, Dissimilarity, Homogeneity, Correlation, ASM |
| Geometric Features | Area, Perimeter, Circularity, Centroid, Extent, Solidity, Eccentricity |
| Labeling Strategy | Urban vs. Rural via per-image Canny edge density — global median threshold (balanced 50/50 split) |
| Classification | Traditional ML classifier trained on 18-dimensional feature vectors |

**Key outputs:** `feature_vectors_full.csv`, `glcm_features.csv`, `geometric_features.csv`, classification accuracy report

---

## Final Project — Deep Vision Pipeline

**Dataset:** All 60 scenes (~1,400+ images), 70/15/15 scene-level train/val/test split

### Architecture

```
ResNet-18 (ImageNet pretrained)
        ↓
PatchTransformerBlock
  Multi-Head Self-Attention over 8×8 spatial tokens
  Captures long-range dependencies (building clusters vs open fields)
        ↓
Global Average Pool → Dropout(0.5) → FC(2)
        ↓
Urban / Rural
```

> **Research Challenge:** Transformer Integration — PatchTransformerBlock applies multi-head self-attention over CNN spatial feature patches for voxel-to-voxel dependency modeling.

| Component | Detail |
|---|---|
| Backbone | ResNet-18 pretrained on ImageNet |
| Transformer | PatchTransformerBlock (MHSA over 8×8 spatial tokens from final feature map) |
| Training | Early stopping, WeightedRandomSampler for class balance |
| Evaluation | Confusion matrix, precision, recall, F1-score, per-class accuracy |
| A1 Integration | Gaussian anti-alias + dtype-aware normalization inside PyTorch `__getitem__` |
| A2 Integration | Canny masking + morphological cleaning on representative images |
| A3 Integration | GLCM feature alignment validation against CNN predictions |

**Key outputs:** Training curves, confusion matrix, qualitative prediction grids, `classification_report.csv`

---

## Dataset

**SpaceNet-7: Multi-Temporal Urban Development**  
Kaggle: [https://www.kaggle.com/datasets/amerii/spacenet-7-multitemporal-urban-development](https://www.kaggle.com/datasets/amerii/spacenet-7-multitemporal-urban-development)

- Multi-temporal satellite imagery across 60 global scenes
- 4-band GeoTIFF format (uint16, ~512×512 pixels per image)
- ~23 monthly timestamps per scene → ~1,400+ total images

> The dataset is not included in this repository. Download it from the link above and update the `ROOT_DIR` path in each notebook before running.

---

## Dependencies

```bash
pip install rasterio scikit-image tqdm opencv-python pandas matplotlib scipy scikit-learn seaborn torch torchvision einops
```

All notebooks are designed to run on **Google Colab** with Google Drive mounted.

---

## Authors

- **Moaz Hassan Khan Manj** (231168)
- **Mahad Jokhio** (231242)

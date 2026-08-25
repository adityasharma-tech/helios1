# HELIOS — Lunar Image Registration & Correspondence

> **Multi-modal, Sun-angle and scale-invariant image correspondence and sub-pixel registration for lunar imagery**

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![Kornia](https://img.shields.io/badge/Kornia-Differentiable%20CV-orange.svg)](https://kornia.github.io/)

## Overview

**HELIOS** is a computer-vision and deep-learning pipeline designed to automatically find reliable correspondences between images of the **same lunar region** acquired under different imaging conditions.

The system addresses three major challenges in lunar image registration:

* ☀️ **Illumination variation** — different Sun azimuth/elevation and resulting shadows
* 📐 **Scale variation** — different spacecraft altitudes and sensor resolutions
* 🛰️ **Viewpoint variation** — different camera positions and viewing geometries

The objective is to transform the source image into the reference-image coordinate system and refine the alignment to **sub-pixel accuracy**, while providing quantitative confidence metrics.

---

# Problem Statement

### SIH Problem Statement — 26166

**Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)**

Traditional image matching becomes unreliable when the same lunar terrain is captured:

* by different sensors,
* at different spatial resolutions,
* from different viewpoints,
* under different illumination conditions.

HELIOS approaches the problem as a **coarse-to-fine registration pipeline** rather than relying on direct pixel-to-pixel comparison.

---

# Core Idea

The system follows a simple principle:

> **First determine where to search, then determine what matches, verify that the matches are geometrically valid, and finally refine the alignment to sub-pixel precision.**

```text
Lunar Images
     │
     ▼
Data Ingestion
     │
     ▼
Geo-Localization
     │
     ▼
Image Pyramid + Tiling
     │
     ▼
Candidate Regions
     │
     ▼
Radiometric Preprocessing
     │
     ▼
DISK / SIFT
     │
     ▼
LightGlue / FLANN
     │
     ▼
Match Filtering
     │
     ▼
USAC-MAGSAC
     │
     ▼
Geometric Transformation
     │
     ▼
Image Warping
     │
     ▼
Residual Alignment
     │
     ▼
Phase Correlation
     │
     ▼
Sub-Pixel Refinement
     │
     ▼
Evaluation & Registered Output
```

---

# System Architecture

## 1. Data Ingestion

Planetary image products and their associated metadata are loaded into the processing pipeline.

The metadata provides information required to interpret the image, including spatial and acquisition-related information.

**Technologies:** Python, NumPy, PDS3/PDS4 handling, GDAL/GeoTIFF where applicable.

---

## 2. Geo-Localization

Before expensive feature matching, HELIOS uses available spatial information to identify where the images overlap.

### Geographic Prior

A geographic prior narrows the search area using known spatial information.

### Overlap Search

The system identifies the region that is potentially common to both images.

This converts:

```text
Search the entire image
```

into:

```text
Search only the likely overlapping region
```

---

# 3. Multi-Scale Search

Large lunar images can contain extremely large numbers of pixels.

HELIOS therefore uses:

### Image Pyramid

Multiple representations of the image are created at different resolutions.

```text
Low Resolution
      ↓
Medium Resolution
      ↓
High Resolution
      ↓
Original Resolution
```

### Pyramid Tiling

Large images are divided into manageable tiles at different pyramid levels.

This enables **coarse-to-fine candidate discovery** without processing the entire image at maximum resolution.

---

# 4. Radiometric Preprocessing

Images acquired under different illumination conditions may have substantially different intensity distributions.

The preprocessing stage can include:

* invalid-pixel handling
* intensity normalization
* percentile-based clipping
* CLAHE
* resolution normalization

The objective is to make structural information easier to compare while reducing undesirable radiometric differences.

---

# 5. Feature Extraction

HELIOS uses learned and classical feature extraction approaches.

### DISK

**DISK** identifies distinctive keypoints and generates learned feature descriptors.

These features represent locally recognizable structures in the lunar surface.

### SIFT

**SIFT** provides a classical feature extraction pathway and fallback/baseline approach.

---

# 6. Feature Matching

Extracted features from the source and reference regions are compared.

### LightGlue

LightGlue performs learned feature correspondence and determines which detected features are likely to represent the same physical location.

### FLANN

FLANN provides a classical, efficient nearest-neighbor matching mechanism and fallback pathway.

The result is a collection of:

```text
Source Point  →  Reference Point
```

correspondences.

---

# 7. Match Filtering

Not every detected correspondence is correct.

The filtering stage removes weak or unlikely matches before geometric estimation.

The remaining matches become candidates for robust geometric verification.

---

# 8. Robust Geometric Verification

HELIOS uses:

### USAC-MAGSAC

USAC-MAGSAC robustly estimates the geometric relationship while identifying inconsistent correspondences.

Conceptually:

```text
Candidate Matches
       │
       ▼
   MAGSAC
       │
 ┌─────┴─────┐
 ▼           ▼
Inliers    Outliers
```

**Inliers** are matches that agree with the estimated geometry.

**Outliers** are inconsistent correspondences.

---

# 9. Geometric Transformation

Once reliable correspondences have been established, the system estimates the transformation required to align the source image with the reference.

Depending on the registration model, this may involve:

* translation
* rotation
* scaling
* affine transformation
* projective transformation / homography

For a projective model:

[
\mathbf{x}' \sim H\mathbf{x}
]

where (H) represents the estimated geometric transformation.

---

# 10. Image Warping

The estimated transformation is applied to the source image.

```text
Source Image
     │
     ▼
Transformation
     │
     ▼
Warp
     │
     ▼
Approximately Registered Image
```

The goal is to bring corresponding lunar structures into the same coordinate system.

---

# 11. Residual Alignment

Even after geometric registration, a small alignment error may remain.

HELIOS therefore performs a final local refinement stage.

---

# 12. Phase Correlation

Phase correlation estimates small residual translations between image regions.

It is particularly useful after the large-scale geometric transformation has already been applied.

The objective is to determine tiny remaining shifts that may be smaller than one pixel.

---

# 13. Sub-Pixel Refinement

The final registration is refined beyond whole-pixel precision.

For example, instead of simply estimating:

```text
Shift = 1 pixel
```

the system can estimate a fractional displacement such as:

```text
Shift = 0.37 pixel
```

This enables the desired **sub-pixel registration accuracy**.

---

# Evaluation

HELIOS evaluates registration quality using multiple complementary metrics.

### RMSE

Measures the magnitude of registration error.

[
RMSE =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}e_i^2
}
]

Lower RMSE indicates better alignment.

### Inlier Ratio

[
Inlier\ Ratio =
\frac{N_{inliers}}
{N_{matches}}
]

A higher ratio indicates that a larger fraction of candidate matches are geometrically consistent.

### SSIM

Measures structural similarity between registered image regions.

### PSNR

Measures image similarity based on reconstruction error.

### Match Uniformity

Measures whether reliable correspondences are distributed across the image rather than concentrated in a small area.

---

# Technology Stack

## Core

* **Python 3.x**
* **NumPy**
* **OpenCV**

## Deep Learning / Computer Vision

* **PyTorch**
* **Kornia**
* **DISK**
* **LightGlue**
* **SIFT**
* **FLANN**

## Geospatial / Planetary Data

* **PDS3 / PDS4**
* **GDAL**
* **GeoTIFF**
* **NumPy `memmap`**

## Registration

* **USAC-MAGSAC**
* **Homography / geometric transformation**
* **Image pyramids**
* **Pyramid tiling**
* **Phase correlation**
* **Sub-pixel refinement**

## CLI / Developer Experience

* **Typer**
* **Rich**

---

# Radiometric Normalization — Experimental Module

The project also includes a deep-learning-based radiometric normalization direction using:

* **U-Net Generator**
* **PatchGAN Discriminator**
* **PyTorch**

The purpose is to reduce appearance differences caused by illumination and sensor characteristics before correspondence estimation.

This module should be considered **experimental/planned unless it has been fully integrated and validated in the current execution pipeline**.

---

# Engineering Design

HELIOS is designed for large planetary imagery where loading an entire image into RAM may be impractical.

### Memory Mapping

`NumPy memmap` allows large image arrays to be accessed without requiring the complete dataset to reside in RAM simultaneously.

### Tiling

Large images are divided into smaller processing regions.

### Coarse-to-Fine Search

Expensive feature extraction and matching are concentrated on promising candidate regions.

Together, these techniques reduce unnecessary computation and memory consumption.

---

# Output

HELIOS is designed to produce:

### 1. Registered Image

The source image transformed into the reference coordinate system.

### 2. Corresponding Match Points

Reliable source/reference point pairs.

```text
(x₁, y₁) → (x₁', y₁')
(x₂, y₂) → (x₂', y₂')
...
```

### 3. Geometric Transformation

Estimated transformation matrix/model.

### 4. Registration Metrics

```text
RMSE
Inlier Count
Inlier Ratio
SSIM
PSNR
Match Uniformity
```

### 5. Confidence / Failure Information

The system should be able to distinguish between:

```text
Reliable Registration
        vs
Insufficient / Unreliable Correspondence
```

rather than blindly returning an alignment.

---

# Why HELIOS?

Traditional registration approaches can struggle when:

```text
Sensor changes
      +
Sun-angle changes
      +
Scale changes
      +
Viewpoint changes
      +
Huge image sizes
```

occur simultaneously.

HELIOS addresses these challenges through a layered approach:

```text
Geographic Prior
       ↓
Search-Space Reduction
       ↓
Pyramid + Tiling
       ↓
Radiometric Preparation
       ↓
Learned Correspondence
       ↓
Robust Geometry
       ↓
Fine Registration
       ↓
Sub-Pixel Refinement
       ↓
Quantitative Validation
```

---

# Project Philosophy

HELIOS is designed around three principles:

### 🎯 Accuracy

Reliable correspondences and sub-pixel refinement.

### ⚡ Efficiency

Geographic priors, pyramid search, tiling, and memory-efficient processing.

### 🛡️ Reliability

Multiple stages of filtering and geometric validation prevent unreliable matches from being blindly accepted.

---

# Future Extensions

Potential future development areas include:

* broader cross-sensor registration across OHRC, TMC and IIRS
* LRO-based reference integration
* DEM-aware geometric correction
* improved illumination normalization
* larger-scale benchmarking
* GPU optimization
* automated confidence estimation
* expanded multi-resolution planetary datasets

---

# Project Structure

A suggested project organization:

```text
helios/
│
├── main.py
│
├── src/
│   ├── registration.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── matching.py
│   ├── geometry.py
│   ├── refinement.py
│   ├── metrics.py
│   ├── tiling.py
│   └── data_ingestion.py
│
├── models/
│   └── gan.py
│
├── data/
│   ├── source/
│   └── reference/
│
├── outputs/
│   ├── registered/
│   ├── matches/
│   └── metrics/
│
├── requirements.txt
└── README.md
```

---

# Quick Conceptual Example

Given:

```text
Source:
Chandrayaan-2 lunar image

Reference:
Lunar reference image
```

HELIOS performs:

```text
1. Read image + metadata
          ↓
2. Determine spatial overlap
          ↓
3. Generate candidate regions
          ↓
4. Normalize image appearance
          ↓
5. Detect features
          ↓
6. Match features
          ↓
7. Remove incorrect matches
          ↓
8. Estimate geometric transformation
          ↓
9. Warp source image
          ↓
10. Refine residual shift
          ↓
11. Achieve sub-pixel registration
          ↓
12. Calculate accuracy metrics (RMSE,Inlier Ratio, SSIM)


```

The final result is a **registered lunar image with corresponding control points and quantitative registration metrics**.

---

# Project Goal

> **To build a robust, efficient and accurate software system capable of finding reliable correspondences between multi-modal lunar images acquired under different scale, viewpoint and illumination conditions, and registering them to sub-pixel accuracy.**

---

## Status

**Development Stage:** Prototype / Research & Development

The pipeline is being developed incrementally, with the core registration path focused on:

**data ingestion → geographic localization → preprocessing → feature extraction → feature matching → robust geometric verification → warping → phase-correlation refinement → evaluation.**

---

## License

Add the project's chosen license here, for example:

```text
MIT License
```

if the team decides to release the implementation under MIT.

---

### Suggested README tagline

> **HELIOS — Connecting different views of the Moon, pixel by pixel.**

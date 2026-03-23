# Beyond Alignment: Cross-Modal Disagreement as Semantic Signal

**CS6240 Multimedia Analysis — BNI Paper**  
**Akshaya Vajpeyarr · National University of Singapore**

---

## Overview

This repository contains the experimental code for the paper:

> **Beyond Alignment: Treating Cross-Modal Disagreement and Absence as Semantic Signals in Multimodal Representation Learning**

The paper argues that cross-modal disagreement $D$ — measured as pairwise embedding divergence — is a meaningful signal that alignment-centric multimodal systems suppress. We provide three controlled experiments using CLIP ViT-B/32 on Flickr30k to support this claim.

---

## Repository Structure

```
.
├── exp1_hallucination_prediction.py   # Experiment 1: confidence–disagreement decoupling
├── exp2_graded_disagreement.py        # Experiment 2: graded mismatch across 6 levels
├── exp3_calibration_gap.py            # Experiment 3: ECE calibration analysis
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## Experiments

### Experiment 1 — Confidence–Disagreement Decoupling
**Question:** Does model confidence reliably track $D$?

Computes $D$ (cosine similarity between image and true-caption embeddings) and model confidence (softmax over $K=15$ candidates) for $N=1{,}000$ Flickr30k pairs. Fits logistic regression predictors and plots ROC curves comparing $D$ alone, confidence alone, and $D$ + confidence combined.

**Key finding:** Confidence does not reliably track $D$ ($r = 0.386$). Many samples with moderate $D$ values (0.20–0.25) receive confidence near 1.0, demonstrating decoupling between geometric disagreement and model certainty.

**Outputs:** `bni_results/exp1_roc.png`, `bni_results/exp1_summary.txt`, `bni_results/exp1_raw.csv`

---

### Experiment 2 — Graded Disagreement Across Perturbation Levels
**Question:** Does $D$ track semantic mismatch more strongly than confidence?

Constructs six perturbation levels per image ($N=500$):

| Level | Condition |
|-------|-----------|
| L0 | Matched (true caption) |
| L1 | Alternative caption for same image |
| L2 | Caption from different image sharing ≥2 content words |
| L3 | Caption from different image with 0 shared content words |
| L4 | Fully random caption |
| L5 | Empty string (missing modality) |

Computes $D$ and confidence at each level, reports Spearman $\rho$ for both signals against mismatch level.

**Key finding:** $D$ tracks mismatch monotonically (Spearman $\rho = -0.577$, $p < 0.001$), more strongly than confidence ($\rho = -0.548$). The null condition (L5) produces $D = 0.199$, correctly intermediate between matched ($0.322$) and random ($0.076$), while confidence anomalously overestimates absence ($0.586$).

**Outputs:** `bni_results/exp2_graded.png`, `bni_results/exp2_boxplot.png`, `bni_results/exp2_summary.txt`, `bni_results/exp2_levels.csv`

---

### Experiment 3 — Calibration Gap (ECE)
**Question:** Does adding $D$ reduce Expected Calibration Error?

Computes ECE for raw confidence, confidence + $D$ blend, and logistic regression combining both signals. Note: retrieval accuracy is near-ceiling (98.7%), making ECE comparison largely degenerate. Results included for completeness; not used as a primary claim in the paper.

**Outputs:** `bni_results/exp3_calibration.png`, `bni_results/exp3_summary.txt`, `bni_results/exp3_ece.csv`

---

## Setup

### Requirements
- Python 3.10 or 3.12
- Apple M3 / MPS backend (or CPU fallback)
- ~3 GB disk space (CLIP weights + Flickr30k)

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/bni-disagreement-experiments.git
cd bni-disagreement-experiments

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (install torch first separately)
pip install --upgrade pip
pip install torch==2.2.2 torchvision==0.17.2
pip install "numpy<2"
pip install open_clip_torch transformers datasets \
            Pillow scipy scikit-learn matplotlib seaborn pandas tqdm
```

### Dataset

The experiments use **Flickr30k**, available via Kaggle:

```bash
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('hsankesara/flickr-image-dataset')"
```

Or download manually from: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

Once downloaded, set `DATASET_PATH` at the top of each experiment script to point to the folder containing `results.csv` and the `flickr30k_images/` subfolder. The default path in each script is:

```python
DATASET_PATH = "/Users/aksha/.cache/kagglehub/datasets/hsankesara/flickr-image-dataset/versions/1/flickr30k_images"
```

---

## Running

```bash
source venv/bin/activate

python exp1_hallucination_prediction.py   # ~5–8 min on M3
python exp2_graded_disagreement.py        # ~8–12 min on M3
python exp3_calibration_gap.py            # ~5–8 min on M3
```

Results are saved to `bni_results/` in the working directory (created automatically).

---

## Results Summary

| Metric | Value |
|--------|-------|
| Retrieval accuracy (Exp 1) | 98.7% |
| Corr($D$, confidence) | $r = 0.386$ |
| Spearman $\rho$: $D$ vs mismatch level | $-0.577$ ($p < 0.001$) |
| Spearman $\rho$: confidence vs mismatch level | $-0.548$ ($p < 0.001$) |
| $D$ at matched (L0) | 0.322 |
| $D$ at null/missing (L5) | 0.199 |
| $D$ at random (L4) | 0.076 |
| Confidence at null (L5) | **0.586** (anomalously high) |
| Confidence at random (L4) | 0.069 |

The null-modality confidence anomaly (0.586 vs 0.069 for random) is the central empirical finding: alignment-centric models conflate absence with agreement, while $D$ correctly encodes it as an intermediate epistemic state.

---

## Model and Data

| Component | Details |
|-----------|---------|
| Vision-language model | CLIP ViT-B/32 (`laion2b_s34b_b79k` weights via `open_clip`) |
| Dataset | Flickr30k (test split, $N = 1{,}000$ / $500$ depending on experiment) |
| Distractor pool | $K = 15$ per retrieval task |
| Hardware | Apple M3 (MPS backend) |
| Seed | 42 (fixed across all experiments) |

---

## Acknowledgements

Experimental code written for CS6240 Multimedia Analysis, NUS.  
AI assistance (GPT-5.2 and Claude) was used for code structure, debugging, and documentation. All experimental design decisions, interpretations, and paper conclusions are my own.

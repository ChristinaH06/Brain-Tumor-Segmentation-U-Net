# 🧠 Brain Tumor Segmentation with U-Net (LGG MRI)

This project implements a **2D U-Net** semantic segmentation model for **brain tumor segmentation** on MRI slices from the **LGG (Lower Grade Glioma)** dataset. The goal is to segment tumor regions from grayscale MRI images using deep learning. 🚀

---

## ✨ 1. Project Overview

- **Task:** Binary semantic segmentation (tumor vs. background)
- **Model:** U-Net (2D)
- **Framework:** PyTorch
- **Dataset:** LGG MRI segmentation dataset (TCGA, Kaggle)
- **Input:** 2D grayscale MRI slices (256 × 256)
- **Output:** Binary tumor mask

---

## 🧠 2. Dataset

The dataset consists of paired MRI slices and binary segmentation masks. Many slices contain **no tumor**, which causes severe class imbalance.

### 🧩 2.1 Class Imbalance Handling
To mitigate this issue:
- **Training set:** slices with empty masks are removed ✅
- **Validation & Test sets:** all slices are retained ✅

This ensures the model learns meaningful tumor features while evaluation remains unbiased.

### 🔗 2.2 Dataset Source
- Kaggle: [LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

---

## 🏗 3. Model Architecture (U-Net)

We use a standard U-Net with an encoder–decoder structure and skip connections:
- Encoder (downsampling)
- Bottleneck
- Decoder (upsampling + skip connections)

Model configuration:
- **Input channels:** 1 (grayscale MRI)
- **Output channels:** 1 (binary mask)

Implementation:
- `model/unet.py`

---

## ⚙️ 4. Training & Evaluation

### 🧪 4.1 Training Setup
- **Loss:** BCE + Dice Loss (`utils/metrics.py`)
- **Optimizer:** AdamW
- **Initial LR:** 3e-4
- **LR Scheduler:** ReduceLROnPlateau
- **Early Stopping:** enabled (patience-based)
- **Split:** 80% / 10% / 10%
- **Training seed:** 42

Hardware note:
- Training/testing can run on **Apple MPS** (Mac) if available, otherwise CPU/CUDA. 🍎⚡️

### 📏 4.2 Metrics
We report the Dice coefficient:
- **Dice (all):** over all test slices
- **Dice (tumor-only):** only on slices with tumors (more clinically meaningful)

---

## ✅ 5. Final Results (Using Provided Pretrained Model)

From `results/test_metrics.txt`:
- **Test Dice (all):** 0.7052  
- **Test Dice (tumor-only):** 0.8656  

📌 Reproducibility note:
- If you **train from scratch**, results may vary slightly across devices (MPS/CPU/CUDA) due to numerical differences.
- Test **visualization sample selection** is reproducible when the same dataset ordering and fixed random seed are used.

---

# 📌 REQUIRED BY COURSE

## 🧰 1) Requirements: Software

### Environment
- **Python:** 3.9+ recommended
- **PyTorch:** 2.x recommended

### Install Dependencies
**Option A (recommended):** install from `requirements.txt`
```bash
pip install -r requirements.txt
```
Option B (manual install):
```bash
pip install torch torchvision numpy opencv-python matplotlib tqdm
```


⸻

## 📦 2) Pretrained Models

### We provide the trained checkpoint:
	•	pretrained/unet_lgg.pth

This allows reproducing the reported test results without training.
- ✅ Because the .pth file is large (about 118M), the repository use Git LFS. 🧳

⸻

## 🧾 3) Preparation for Testing

### 3.1 Expected Directory Structure

Before running `train.py` or `test.py`, prepare the dataset in:

```text
data/
  images/
    img_0001.png
    img_0002.png
    ...
  masks/
    img_0001.png
    img_0002.png
    ...
```

**Rules**
- Each `data/images/<name>.png` must have a corresponding `data/masks/<name>.png`
- Masks should be binary: **0 = background**, **1 = tumor**

### 3.2 Data Format Notes
- Images are loaded as grayscale
- All images/masks are resized to **256×256** internally
- If your dataset is not already `.png` pairs with identical names, preprocess/rename first

(Optional) preprocessing script (if needed):
- `prepare_dataset.py`

⸻

## ▶️ 6. Commands to Run (Training + Testing)

### 6.1 Train

Run:
```bash
python train.py
```

Output:
- Best checkpoint saved to: `pretrained/unet_lgg.pth`

### 6.2 Test (Evaluate + Save Visualizations)
```bash
python test.py
```

Outputs:
- We got `results/test_metrics.txt` and `results/testviz_*.png`

⸻

## 🖼 7. Qualitative Results

### Visualization outputs saved in results/ include:
- Input MRI 
- Gund truth mask
- Predicted mask
- Overlay visualization 🪄

⸻

## 👥 8. Member Contributions
- Yaqi Huang (1220013086): Implemented and verified the U-Net model architecture (`model/unet.py`), contributed to training pipeline design (`train.py`) including `optimizer/scheduler/early-stopping` settings, organized experimental outputs (`results/`), and participated in experiment setup and debugging.
- Meiyi Qian (1220008758): Implemented dataset loading and preprocessing utilities (`utils/dataset.py`) including the training-only empty-mask filtering strategy, implemented evaluation and visualization (`test.py`) and metric functions (`utils/metrics.py`), and drafted the final report and GitHub project documentation (`README requirements`, `testing instructions`).

⸻

## 🙏 9. Acknowledgments
- Dataset provided by the LGG MRI segmentation dataset (TCGA) via Kaggle
- U-Net inspired by the original U-Net paper for biomedical image segmentation 📚


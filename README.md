**Brain Tumor Segmentation with U-Net (LGG MRI)**
This project implements a U-Net–based semantic segmentation model for brain tumor segmentation on MRI slices from the LGG (Lower Grade Glioma) dataset.
The goal is to accurately segment tumor regions from grayscale MRI images using deep learning.

📌 Project Overview
Task: Binary semantic segmentation (tumor vs. background)
Model: U-Net (2D)
Framework: PyTorch
Dataset: LGG MRI segmentation dataset
Input: 2D grayscale MRI slices (256 × 256)
Output: Binary tumor mask

🧠 Dataset
The dataset consists of paired MRI slices and binary segmentation masks.
Each image has a corresponding mask with pixel values:
0: background
1: tumor region
Many MRI slices contain no tumor, leading to severe class imbalance.
Handling Class Imbalance
To mitigate this issue:
Training set: slices with empty masks were removed
Validation & Test sets: all slices were retained
This ensures that the model learns meaningful tumor features while evaluation remains unbiased.

🏗 Model Architecture
The segmentation model is a standard U-Net, consisting of:
Encoder (downsampling path)
Bottleneck
Decoder (upsampling path with skip connections)
Input channels: 1 (grayscale MRI)
Output channels: 1 (binary mask)

⚙️ Training Details
Loss Function: BCE + Dice Loss
Optimizer: AdamW
Initial Learning Rate: 3e-4
Scheduler: ReduceLROnPlateau
Early Stopping: Enabled
Train / Val / Test split: 80% / 10% / 10%
Random seed (training): 42
Training was performed on CPU due to hardware constraints.

📊 Evaluation Metrics
Performance is evaluated using the Dice coefficient:
Dice (all): computed over all test slices
Dice (tumor-only): computed only on slices containing tumors
Reporting tumor-only Dice better reflects segmentation quality on clinically relevant regions.

✅ Final Results
Test Set Performance:
Dice (all):        0.6806
Dice (tumor-only): 0.8576
These results demonstrate strong tumor segmentation performance despite severe class imbalance.

🖼 Qualitative Results
Qualitative results are visualized using:
Input MRI
Ground truth mask
Predicted mask
Overlay visualization
Test images are randomly selected using a fixed random seed to ensure reproducibility.
Random seed (test visualization): 56
Example visualizations can be found in the results/ directory.

▶️ How to Run
1. Install Dependencies
pip install torch torchvision numpy opencv-python matplotlib tqdm
2. Train the Model
python train.py
The best model will be saved to:
pretrained/unet_lgg.pth
3. Evaluate on Test Set
python test.py --seed 56
Test metrics will be saved to:
results/test_metrics.txt

📝 Notes
All random seeds are fixed for reproducibility.
Test visualization seed only affects displayed samples and does not alter evaluation metrics.
The project follows standard academic practices for medical image segmentation.

📎 Acknowledgments
Dataset provided by the LGG MRI segmentation dataset (TCGA).
U-Net architecture inspired by the original U-Net paper for biomedical image segmentation.

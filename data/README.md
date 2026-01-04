# Dataset Note (LGG MRI Segmentation)

This folder is reserved for dataset-related files.  
**To keep the repository lightweight, the raw dataset and the generated image/mask files are NOT included in this repo.**

## Dataset Source
We use the **LGG (Lower Grade Glioma) MRI Segmentation Dataset** from **Kaggle**.  
<https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation>

## Expected Local Structure
After preparation, your local project should look like:

```text
Brain-Tumor-Segmentation-U-Net/
├── data/
│   ├── images/   # Generated 2D MRI slices (.png)
│   └── masks/    # Generated binary masks (.png)
├── raw_data/     # Original downloaded Kaggle files
└── ...
```

## How to Prepare the Dataset
1) Download the Kaggle LGG dataset and place the raw files under a folder such as:
- `raw_data/`  (recommended)

2) Run the dataset preparation script:
```bash
python prepare_dataset.py
```

###This script will generate paired files under:
- data/images/*.png
- data/masks/*.png

###Notes
- Many slices contain empty masks (no tumor). Our training script filters empty-mask samples only in the training set to reduce class imbalance.
- Validation and test sets keep all samples for unbiased evaluation.

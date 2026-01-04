import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from model.unet import UNet
from utils.dataset import LGGSegmentationDataset, SubsetWithFilter
from utils.metrics import dice_coefficient_tensor


def set_seed(seed=56):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_indices(n_total, seed=56):
    rng = np.random.default_rng(seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)

    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def evaluate(model, loader, device):
    model.eval()
    dices_all = []
    dices_tumor = []

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()

            d_all = dice_coefficient_tensor(preds, masks).item()
            dices_all.append(d_all)

            tumor_present = (masks.sum(dim=(1, 2, 3)) > 0)
            if tumor_present.any():
                d_tumor = dice_coefficient_tensor(preds[tumor_present], masks[tumor_present]).item()
                dices_tumor.append(d_tumor)

    dice_all = float(np.mean(dices_all)) if len(dices_all) else 0.0
    dice_tumor = float(np.mean(dices_tumor)) if len(dices_tumor) else 0.0
    return dice_all, dice_tumor


def save_visualizations(model, dataset, device, out_dir="results", num_samples=8):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    # pick indices: prefer tumor samples, but include some empty too
    tumor_indices = []
    empty_indices = []
    for i in range(len(dataset)):
        _, mask, _ = dataset[i]
        if mask.sum().item() > 0:
            tumor_indices.append(i)
        else:
            empty_indices.append(i)

    chosen = []
    if len(tumor_indices) > 0:
        chosen += random.sample(tumor_indices, k=min(num_samples // 2, len(tumor_indices)))
    if len(empty_indices) > 0 and len(chosen) < num_samples:
        chosen += random.sample(empty_indices, k=min(num_samples - len(chosen), len(empty_indices)))

    for k, idx in enumerate(chosen, start=1):
        image, mask, fname = dataset[idx]
        x = image.unsqueeze(0).to(device)  # [1,1,H,W]
        gt = mask.squeeze(0).numpy()

        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
            pred = (prob > 0.5).astype(np.float32)

        img = image.squeeze(0).numpy()

        # plot: input / GT / pred / overlay
        fig = plt.figure(figsize=(12, 4))

        ax1 = fig.add_subplot(1, 4, 1)
        ax1.imshow(img, cmap="gray")
        ax1.set_title("Input MRI")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 4, 2)
        ax2.imshow(gt, cmap="gray")
        ax2.set_title("Ground Truth")
        ax2.axis("off")

        ax3 = fig.add_subplot(1, 4, 3)
        ax3.imshow(pred, cmap="gray")
        ax3.set_title("Prediction")
        ax3.axis("off")

        ax4 = fig.add_subplot(1, 4, 4)
        ax4.imshow(img, cmap="gray")
        ax4.imshow(pred, alpha=0.4)
        ax4.set_title("Overlay")
        ax4.axis("off")

        out_path = os.path.join(out_dir, f"testviz_{k}_{fname}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print("Saved:", out_path)


def main():
    set_seed(56)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # data & split (same seed/same logic as train.py)
    base_ds = LGGSegmentationDataset("data/images", "data/masks", resize=(256, 256))
    train_idx, val_idx, test_idx = split_indices(len(base_ds), seed=56)

    # test keeps all samples
    test_ds = SubsetWithFilter(base_ds, test_idx, filter_empty=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    # load model
    model = UNet(in_channels=1, out_channels=1).to(device)
    model_path = "pretrained/unet_lgg.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Loaded:", model_path)

    # evaluate
    dice_all, dice_tumor = evaluate(model, test_loader, device)
    print(f"Test Dice (all): {dice_all:.4f}")
    print(f"Test Dice (tumor-only): {dice_tumor:.4f}")

    # save metrics
    os.makedirs("results", exist_ok=True)
    with open("results/test_metrics.txt", "w") as f:
        f.write(f"Test Dice (all): {dice_all:.4f}\n")
        f.write(f"Test Dice (tumor-only): {dice_tumor:.4f}\n")
    print("Saved: results/test_metrics.txt")

    # save some visualizations
    save_visualizations(model, test_ds, device, out_dir="results", num_samples=10)


if __name__ == "__main__":
    main()

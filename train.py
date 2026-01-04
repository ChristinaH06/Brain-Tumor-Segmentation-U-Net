import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.unet import UNet
from utils.dataset import LGGSegmentationDataset, SubsetWithFilter
from utils.metrics import bce_dice_loss, dice_coefficient_tensor


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_indices(n_total, seed=42):
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

    val_dice_all = float(np.mean(dices_all)) if len(dices_all) else 0.0
    val_dice_tumor = float(np.mean(dices_tumor)) if len(dices_tumor) else 0.0
    return val_dice_all, val_dice_tumor


def main():
    set_seed(42)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    # ---- data ----
    base_ds = LGGSegmentationDataset("data/images", "data/masks", resize=(256, 256))
    train_idx, val_idx, test_idx = split_indices(len(base_ds), seed=42)

    train_ds = SubsetWithFilter(base_ds, train_idx, filter_empty=True)   # only train filters empty
    val_ds = SubsetWithFilter(base_ds, val_idx, filter_empty=False)      # keep all
    test_ds = SubsetWithFilter(base_ds, test_idx, filter_empty=False)    # keep all (Phase 5 use)

    # CPU: batch size不要太大，否则每步更慢；8/16二选一
    batch_size = 8

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ---- model ----
    model = UNet(in_channels=1, out_channels=1).to(device)

    # AdamW通常比Adam更稳一点
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # 验证集不提升就自动降lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    os.makedirs("pretrained", exist_ok=True)

    # ---- training control ----
    epochs = 50                 # CPU慢慢训练的时候就开大一点，因为EarlyStopping会自动停
    early_stop_patience = 6     # 连续6个epoch tumor-dice不提升就停止
    best_score = -1.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = bce_dice_loss(logits, masks, bce_weight=0.7)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        train_loss = float(np.mean(losses)) if len(losses) else 0.0

        val_dice_all, val_dice_tumor = evaluate(model, val_loader, device)
        score_to_save = val_dice_tumor  # 用 tumor-only 作为保存标准
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch} | lr={lr_now:.2e} | train_loss={train_loss:.4f} "
            f"| val_dice_all={val_dice_all:.4f} | val_dice_tumor={val_dice_tumor:.4f}"
        )

        # scheduler based on tumor-only dice
        scheduler.step(score_to_save)

        if score_to_save > best_score:
            best_score = score_to_save
            no_improve = 0
            torch.save(model.state_dict(), "pretrained/unet_lgg.pth")
            print("Saved best model to pretrained/unet_lgg.pth")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"Early stopping triggered. Best val_dice_tumor={best_score:.4f}")
                break

    print("Training done. Best val_dice_tumor:", best_score)
    print("Note: test evaluation + visualization will be done in Phase 5 using test.py.")


if __name__ == "__main__":
    main()

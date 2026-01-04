import os
import random
import matplotlib.pyplot as plt

from utils.dataset import LGGSegmentationDataset

def main():
    images_dir = "data/images"
    masks_dir = "data/masks"
    os.makedirs("results", exist_ok=True)

    ds = LGGSegmentationDataset(images_dir, masks_dir, resize=None)

    # pick 5 random samples
    indices = random.sample(range(len(ds)), 5)

    for i, idx in enumerate(indices):
        image, mask, fname = ds[idx]

        img_np = image.squeeze(0).numpy()
        mask_np = mask.squeeze(0).numpy()

        unique_vals = sorted(list(set(mask_np.flatten().tolist())))
        print(f"{fname} | image={tuple(image.shape)} mask={tuple(mask.shape)} mask_unique={unique_vals[:10]}")

        fig = plt.figure(figsize=(10, 4))

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(img_np, cmap="gray")
        ax1.set_title("MRI Image")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(mask_np, cmap="gray")
        ax2.set_title("Ground Truth Mask")
        ax2.axis("off")

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(img_np, cmap="gray")
        ax3.imshow(mask_np, alpha=0.4)
        ax3.set_title("Overlay")
        ax3.axis("off")

        out_path = os.path.join("results", f"check_{i+1}_{fname}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()

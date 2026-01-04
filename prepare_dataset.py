import random
import shutil
from pathlib import Path

#可改：比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 56

def main():
    root = Path(__file__).resolve().parent
    img_dir = root / "data" / "images"
    mask_dir = root / "data" / "masks"
    out_dir = root / "data" / "split"

    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError("找不到 data/images 或 data/masks，请确认数据路径。")

    #收集所有图片
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    imgs = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not imgs:
        raise RuntimeError("data/images 里没有找到任何图片。")

    #要求 image 和 mask 同名（含后缀也同名）
    pairs = []
    for img in imgs:
        m = mask_dir / img.name
        if m.exists():
            pairs.append((img, m))

    if not pairs:
        raise RuntimeError("没有找到任何 image-mask 对：请确认 masks 里与 images 同名对应。")

    random.seed(SEED)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    #创建目录并复制
    for sp, items in splits.items():
        (out_dir / sp / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / sp / "masks").mkdir(parents=True, exist_ok=True)
        for img, msk in items:
            shutil.copy2(img, out_dir / sp / "images" / img.name)
            shutil.copy2(msk, out_dir / sp / "masks" / msk.name)

    print(f"Total pairs: {n}")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    print(f"Saved to: {out_dir}")

if __name__ == "__main__":
    main()
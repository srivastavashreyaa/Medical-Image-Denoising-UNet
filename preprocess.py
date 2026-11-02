"""Preprocessing utilities + CLI to build paired Clean/Noisy datasets.

Pipeline: grayscale -> resize -> CLAHE -> [0,1] normalisation -> augmentation
-> synthetic noise injection (gaussian / poisson / salt-pepper / mixed).
"""
import argparse
import os

import cv2
import numpy as np

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_gray(path, size=256):
    """Read any image as a grayscale float32 array in [0, 1] of shape (size, size)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def apply_clahe(img_u8, clip=2.0, tile=8):
    """Contrast Limited Adaptive Histogram Equalisation on a uint8 image."""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(img_u8)


def normalize(img_u8):
    return img_u8.astype(np.float32) / 255.0


def preprocess_image(path, size=256):
    """Full clean-image preprocessing: grayscale, resize, CLAHE, normalise."""
    return normalize(apply_clahe(load_gray(path, size)))


def augment(img, rng):
    """Random flip / rotation / brightness jitter on a [0,1] float image."""
    if rng.random() < 0.5:
        img = np.fliplr(img)
    if rng.random() < 0.3:
        img = np.flipud(img)
    angle = rng.uniform(-15, 15)
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    img = np.clip(img * rng.uniform(0.9, 1.1), 0, 1)
    return img.astype(np.float32)


def add_noise(img, kind="mixed", rng=None, sigma=0.08, level=None):
    """Inject synthetic noise into a [0,1] image.

    level=None -> random strength (used for training data).
    level in [0,1] -> fixed strength (used by the demo app's slider).
    """
    rng = rng or np.random.default_rng()
    if kind == "mixed":
        kind = rng.choice(["gaussian", "poisson", "saltpepper"])
    noisy = img.copy()
    if kind == "gaussian":
        s = rng.uniform(0.04, sigma * 1.5) if level is None else 0.02 + 0.18 * level
        noisy = noisy + rng.normal(0, s, img.shape)
    elif kind == "poisson":
        scale = rng.uniform(30, 120) if level is None else 150 - 130 * level  # lower = noisier
        noisy = rng.poisson(np.clip(noisy, 0, 1) * scale) / scale
    elif kind == "saltpepper":
        mask = rng.random(img.shape)
        amount = rng.uniform(0.01, 0.04) if level is None else 0.005 + 0.07 * level
        noisy[mask < amount / 2] = 0.0
        noisy[mask > 1 - amount / 2] = 1.0
    else:
        raise ValueError(f"Unknown noise type: {kind}")
    return np.clip(noisy, 0, 1).astype(np.float32)


def build_dataset(input_dir, output_dir, size=256, n_aug=2, noise="mixed", seed=42):
    rng = np.random.default_rng(seed)
    clean_dir = os.path.join(output_dir, "Clean")
    noisy_dir = os.path.join(output_dir, "Noisy")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(IMG_EXT))
    if not files:
        raise SystemExit(f"No images found in {input_dir}")

    count = 0
    for name in files:
        try:
            base = preprocess_image(os.path.join(input_dir, name), size)
        except ValueError as e:      # skip corrupt / unreadable files instead of crashing
            print("skipping:", e)
            continue
        stem = os.path.splitext(name)[0]
        variants = [base] + [augment(base, rng) for _ in range(n_aug)]
        for i, clean in enumerate(variants):
            noisy = add_noise(clean, noise, rng)
            out = f"{stem}_{i}.png"
            cv2.imwrite(os.path.join(clean_dir, out), (clean * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(noisy_dir, out), (noisy * 255).astype(np.uint8))
            count += 1
    print(f"Wrote {count} Clean/Noisy pairs to {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", required=True, help="folder of raw CT / X-ray images")
    p.add_argument("--output_dir", default="./data")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--augment", type=int, default=2, help="augmented copies per image")
    p.add_argument("--noise_type", default="mixed",
                   choices=["gaussian", "poisson", "saltpepper", "mixed"])
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    build_dataset(a.input_dir, a.output_dir, a.img_size, a.augment, a.noise_type, a.seed)

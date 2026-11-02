"""Save a qualitative figure: clean | noisy | denoised (+ PSNR) for random validation images."""
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import metrics
from inference import denoise


def main(a):
    clean_dir, noisy_dir = os.path.join(a.data_dir, "Clean"), os.path.join(a.data_dir, "Noisy")
    names = sorted(os.listdir(clean_dir))
    names = list(np.random.default_rng(a.seed).choice(names, min(a.n, len(names)), replace=False))
    model = tf.keras.models.load_model(a.model, compile=False)
    fig, ax = plt.subplots(len(names), 3, figsize=(9, 3 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        c = cv2.imread(os.path.join(clean_dir, name), 0)
        n = cv2.imread(os.path.join(noisy_dir, name), 0)
        d = denoise(model, n, a.img_size)
        cf, nf, df = (x.astype(np.float32) / 255 for x in (c, n, d))
        for k, (img, title) in enumerate([(c, "Clean (ground truth)"),
                                          (n, f"Noisy  PSNR {metrics.psnr(cf, nf):.1f} dB"),
                                          (d, f"Denoised  PSNR {metrics.psnr(cf, df):.1f} dB")]):
            ax[r, k].imshow(img, cmap="gray", vmin=0, vmax=255)
            ax[r, k].set_title(title, fontsize=9); ax[r, k].axis("off")
    plt.tight_layout(); plt.savefig(a.output, dpi=150); plt.close()
    print(f"Saved {a.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--model", default="denoising_model.keras")
    p.add_argument("--output", default="results/qualitative_comparison.png")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    main(p.parse_args())

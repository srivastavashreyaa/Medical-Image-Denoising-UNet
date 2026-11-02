"""Denoise a single image or a whole folder with a trained model."""
import argparse
import os

import cv2
import numpy as np
import tensorflow as tf

from preprocess import IMG_EXT


def denoise(model, gray_u8, size=256):
    """gray_u8: 2-D uint8 array. Returns a uint8 denoised image at the original resolution."""
    h, w = gray_u8.shape
    x = cv2.resize(gray_u8, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    y = model.predict(x[None, ..., None], verbose=0)[0, ..., 0]
    y = cv2.resize(y, (w, h), interpolation=cv2.INTER_CUBIC)
    return (np.clip(y, 0, 1) * 255).astype(np.uint8)


def process(model, src, dst, size, compare):
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"skip (unreadable): {src}")
        return
    out = denoise(model, img, size)
    cv2.imwrite(dst, np.hstack([img, out]) if compare else out)
    print(f"{src} -> {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="denoising_model.keras")
    p.add_argument("--input"); p.add_argument("--output", default="denoised.png")
    p.add_argument("--input_dir"); p.add_argument("--output_dir", default="./denoised_results")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--compare", action="store_true", help="save noisy|denoised side by side")
    a = p.parse_args()
    m = tf.keras.models.load_model(a.model, compile=False)
    if a.input:
        process(m, a.input, a.output, a.img_size, a.compare)
    elif a.input_dir:
        os.makedirs(a.output_dir, exist_ok=True)
        for f in sorted(os.listdir(a.input_dir)):
            if f.lower().endswith(IMG_EXT):
                process(m, os.path.join(a.input_dir, f),
                        os.path.join(a.output_dir, os.path.splitext(f)[0] + "_denoised.png"),
                        a.img_size, a.compare)
    else:
        p.error("give --input or --input_dir")

"""Create synthetic lung-like phantom images so the pipeline runs end-to-end
without downloading a dataset. NOT real medical data - use only for a smoke test.
For real results use a public dataset (see README)."""
import argparse
import os

import cv2
import numpy as np


def phantom(size, rng):
    img = np.zeros((size, size), np.float32)
    c = size // 2
    cv2.ellipse(img, (c, c), (int(size * .42), int(size * .46)), 0, 0, 360, 0.55, -1)  # torso
    for dx in (-1, 1):  # lungs
        cx = int(c + dx * size * .19 + rng.integers(-6, 6))
        cv2.ellipse(img, (cx, c), (int(size * .15 + rng.integers(-5, 5)), int(size * .30)),
                    0, 0, 360, 0.12, -1)
    cv2.line(img, (c, int(size * .1)), (c, int(size * .9)), 0.9, int(size * .03))  # spine
    for _ in range(rng.integers(4, 10)):  # nodules / vessels
        x, y = rng.integers(int(size * .2), int(size * .8), 2)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(2, 7)), float(rng.uniform(.4, .8)), -1)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="./raw_images")
    p.add_argument("--count", type=int, default=60)
    p.add_argument("--img_size", type=int, default=256)
    a = p.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(a.count):
        cv2.imwrite(os.path.join(a.output_dir, f"phantom_{i:03d}.png"), phantom(a.img_size, rng))
    print(f"Wrote {a.count} phantom images to {a.output_dir}")

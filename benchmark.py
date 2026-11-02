"""Measure preprocessing throughput: sequential Python loop vs parallel tf.data pipeline.
Use YOUR measured numbers (not guesses) when you describe the speed-up on your resume."""
import argparse
import time

import numpy as np
import tensorflow as tf

import cv2
from train import list_pairs, make_dataset

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default="./data")
p.add_argument("--img_size", type=int, default=256)
p.add_argument("--batch_size", type=int, default=8)
a = p.parse_args()

noisy, clean = list_pairs(a.data_dir)

t0 = time.perf_counter()
for n, c in zip(noisy, clean):                      # baseline: one image at a time, same work
    for path in (n, c):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (a.img_size, a.img_size)).astype(np.float32) / 255.0
seq = time.perf_counter() - t0

ds = make_dataset(noisy, clean, a.img_size, a.batch_size, False)
t0 = time.perf_counter()
for _ in ds:                                         # optimised: parallel + prefetch
    pass
par = time.perf_counter() - t0

print(f"sequential: {seq:.2f}s | tf.data pipeline: {par:.2f}s | time saved: {100*(1-par/seq):.1f}%")

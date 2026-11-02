import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import metrics
from preprocess import add_noise, apply_clahe, augment, normalize


def test_noise_kinds_keep_range_and_shape():
    rng = np.random.default_rng(0)
    img = np.full((64, 64), 0.5, np.float32)
    for kind in ["gaussian", "poisson", "saltpepper", "mixed"]:
        out = add_noise(img, kind, rng)
        assert out.shape == img.shape and out.min() >= 0 and out.max() <= 1
        assert not np.allclose(out, img)


def test_clahe_normalize_augment():
    rng = np.random.default_rng(0)
    u8 = (rng.random((64, 64)) * 255).astype(np.uint8)
    n = normalize(apply_clahe(u8))
    assert n.dtype == np.float32 and 0 <= n.min() and n.max() <= 1
    assert augment(n, rng).shape == n.shape


def test_metrics_improve_when_closer():
    rng = np.random.default_rng(0)
    clean = rng.random((64, 64)).astype(np.float32)
    noisy = np.clip(clean + rng.normal(0, .2, clean.shape), 0, 1).astype(np.float32)
    better = np.clip(clean + rng.normal(0, .05, clean.shape), 0, 1).astype(np.float32)
    assert metrics.psnr(clean, better) > metrics.psnr(clean, noisy)
    assert metrics.ssim(clean, better) > metrics.ssim(clean, noisy)


def test_unet_shapes():
    from model import build_unet
    m = build_unet(size=64, base=8)
    assert m.output_shape == (None, 64, 64, 1)
    assert m.predict(np.zeros((1, 64, 64, 1), np.float32), verbose=0).shape == (1, 64, 64, 1)

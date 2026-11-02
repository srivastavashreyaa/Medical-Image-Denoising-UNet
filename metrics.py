"""Image-quality metrics (inputs: float arrays in [0,1], shape HxW)."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def psnr(ref, img):
    return float(peak_signal_noise_ratio(ref, img, data_range=1.0))


def ssim(ref, img):
    return float(structural_similarity(ref, img, data_range=1.0))


def compare(clean, noisy, denoised):
    """Return before/after metrics of a denoised image against its clean reference."""
    return {
        "psnr_noisy": psnr(clean, noisy), "psnr_denoised": psnr(clean, denoised),
        "ssim_noisy": ssim(clean, noisy), "ssim_denoised": ssim(clean, denoised),
        "mse_noisy": mse(clean, noisy), "mse_denoised": mse(clean, denoised),
    }

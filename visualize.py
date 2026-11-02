"""Plots from denoising_metrics.csv (distributions + before/after scatter)."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main(csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, col, title in zip(ax, ["psnr", "ssim"], ["PSNR (dB)", "SSIM"]):
        a.hist(df[f"{col}_noisy"], bins=20, alpha=.6, label="noisy")
        a.hist(df[f"{col}_denoised"], bins=20, alpha=.6, label="denoised")
        a.set_title(title); a.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "metric_distributions.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(df.psnr_noisy, df.psnr_denoised, s=14)
    lo, hi = df[["psnr_noisy", "psnr_denoised"]].min().min(), df[["psnr_noisy", "psnr_denoised"]].max().max()
    plt.plot([lo, hi], [lo, hi], "r--", label="no change")
    plt.xlabel("PSNR noisy (dB)"); plt.ylabel("PSNR denoised (dB)"); plt.legend()
    plt.title("Per-image improvement"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "improvement_scatter.png"), dpi=150); plt.close()
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="denoising_metrics.csv")
    p.add_argument("--output_dir", default="./results")
    a = p.parse_args()
    main(a.metrics, a.output_dir)

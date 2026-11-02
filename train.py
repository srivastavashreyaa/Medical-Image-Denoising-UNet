"""Train the denoising U-Net on a paired Clean/Noisy dataset (see preprocess.py)."""
import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf

import metrics
from model import build_unet


def list_pairs(data_dir):
    clean_dir, noisy_dir = os.path.join(data_dir, "Clean"), os.path.join(data_dir, "Noisy")
    names = sorted(f for f in os.listdir(clean_dir) if os.path.exists(os.path.join(noisy_dir, f)))
    if not names:
        raise SystemExit(f"No Clean/Noisy pairs found under {data_dir}")
    return [os.path.join(noisy_dir, n) for n in names], [os.path.join(clean_dir, n) for n in names]


def make_dataset(noisy, clean, size, batch, shuffle):
    """Parallel tf.data input pipeline (decode + resize + scale in parallel, prefetched)."""
    def load(n, c):
        def read(p):
            img = tf.io.decode_png(tf.io.read_file(p), channels=1)
            img = tf.image.resize(img, (size, size))
            return tf.cast(img, tf.float32) / 255.0
        return read(n), read(c)

    ds = tf.data.Dataset.from_tensor_slices((noisy, clean))
    if shuffle:
        ds = ds.shuffle(len(noisy), seed=0)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def evaluate(model, noisy, clean, size, out_csv):
    rows = []
    for n, c in zip(noisy, clean):
        ni = tf.io.decode_png(tf.io.read_file(n), channels=1)
        ci = tf.io.decode_png(tf.io.read_file(c), channels=1)
        ni = tf.image.resize(ni, (size, size)).numpy()[..., 0] / 255.0
        ci = tf.image.resize(ci, (size, size)).numpy()[..., 0] / 255.0
        den = model.predict(ni[None, ..., None], verbose=0)[0, ..., 0]
        rows.append({"image": os.path.basename(n), **metrics.compare(ci, ni, den)})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    mse_red = 100 * (1 - df.mse_denoised.mean() / df.mse_noisy.mean())
    print(f"\nValidation on {len(df)} images")
    print(f"  PSNR : {df.psnr_noisy.mean():.2f} -> {df.psnr_denoised.mean():.2f} dB")
    print(f"  SSIM : {df.ssim_noisy.mean():.4f} -> {df.ssim_denoised.mean():.4f}")
    print(f"  MSE reduction: {mse_red:.2f}%")
    return df


def plot_history(hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE loss"); plt.title("Training history"); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--base_filters", type=int, default=32)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset_name", default="unspecified")
    p.add_argument("--info_out", default="training_info.json")
    p.add_argument("--model_out", default="denoising_model.keras")
    p.add_argument("--metrics_out", default="denoising_metrics.csv")
    p.add_argument("--history_out", default="training_history.png")
    a = p.parse_args()

    if tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("GPU detected -> mixed precision enabled")

    noisy, clean = list_pairs(a.data_dir)
    # split by SOURCE image (name without the trailing _<aug index>) so that augmented copies of
    # one scan never end up in both train and validation (that would leak and inflate the metrics)
    groups = [os.path.basename(p).rsplit("_", 1)[0] for p in noisy]
    uniq = sorted(set(groups))
    rng = np.random.default_rng(a.seed)
    val_groups = set(rng.permutation(uniq)[:max(1, int(len(uniq) * a.val_split))])
    val_i = np.array([i for i, g in enumerate(groups) if g in val_groups])
    tr_i = np.array([i for i, g in enumerate(groups) if g not in val_groups])
    print(f"{len(uniq)} source images -> {len(tr_i)} train pairs, {len(val_i)} validation pairs")
    pick = lambda lst, ii: [lst[i] for i in ii]
    train_ds = make_dataset(pick(noisy, tr_i), pick(clean, tr_i), a.img_size, a.batch_size, True)
    val_ds = make_dataset(pick(noisy, val_i), pick(clean, val_i), a.img_size, a.batch_size, False)

    model = build_unet(a.img_size, a.base_filters)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(a.model_out, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
    ]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, callbacks=cbs, verbose=2)
    plot_history(hist, a.history_out)
    pd.DataFrame(hist.history).to_csv(os.path.splitext(a.history_out)[0] + ".csv", index_label="epoch")
    df = evaluate(model, pick(noisy, val_i), pick(clean, val_i), a.img_size, a.metrics_out)
    import datetime, json
    json.dump({"dataset": a.dataset_name, "img_size": a.img_size, "base_filters": a.base_filters,
               "source_images": len(uniq), "train_pairs": int(len(tr_i)), "val_pairs": int(len(val_i)),
               "epochs_run": len(hist.history["loss"]), "batch_size": a.batch_size,
               "trained_on": datetime.date.today().isoformat(),
               "psnr_noisy": float(df.psnr_noisy.mean()), "psnr_denoised": float(df.psnr_denoised.mean()),
               "ssim_noisy": float(df.ssim_noisy.mean()), "ssim_denoised": float(df.ssim_denoised.mean()),
               "mse_reduction_pct": float(100 * (1 - df.mse_denoised.mean() / df.mse_noisy.mean()))},
              open(a.info_out, "w"), indent=2)
    print(f"Saved model -> {a.model_out}")


if __name__ == "__main__":
    main()

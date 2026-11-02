"""CT Denoiser - interactive Streamlit dashboard for the U-Net medical image denoising model.

Tabs: Denoise Image | Analytics Dashboard | Training History | How To Use
Run:  streamlit run app.py
"""
import glob
import io
import json
import os
import zipfile

import cv2
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

import metrics
from generate_demo_data import phantom
from inference import denoise
from preprocess import IMG_EXT, add_noise, apply_clahe

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATES = ["denoising_model.keras", "best_denoising_model.keras",
                    "denoising_model.h5", "demo_output/model.keras"]
UPLOAD_TYPES = [e.strip(".") for e in IMG_EXT]

st.set_page_config(page_title="CT Denoiser", page_icon="🫁", layout="wide")

st.markdown("""
<style>
.hero{padding:2rem 2.2rem;border-radius:22px;border:1px solid rgba(120,130,255,.25);
      background:linear-gradient(135deg,rgba(60,50,150,.35),rgba(20,70,130,.35));margin-bottom:1rem}
.hero h3{margin:0 0 .6rem 0;color:#8f94ff}
.hero p{margin:.15rem 0;font-size:1.05rem;opacity:.85}
.brand{text-align:center;padding:1.2rem 0 .4rem 0}
.brand .logo{font-size:3.6rem;line-height:1}
.brand .name{font-size:1.5rem;font-weight:800;color:#7c83ff;margin-top:.3rem}
.brand .sub{font-size:.8rem;opacity:.7}
.kv{display:flex;justify-content:space-between;gap:.6rem;padding:.42rem 0;font-size:.86rem}
.kv span:first-child{opacity:.65}.kv span:last-child{text-align:right;font-weight:600}
.badge{display:inline-block;padding:.15rem .6rem;border-radius:99px;background:rgba(124,131,255,.18);
       color:#aab0ff;font-size:.78rem;margin-right:.3rem}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- helpers
def first_existing(candidates):
    for c in candidates:
        if os.path.exists(os.path.join(ROOT, c)):
            return c
    return candidates[0]


def find(directory, names):
    for n in names:
        p = os.path.join(directory, n)
        if os.path.exists(p):
            return p
    return None


@st.cache_resource(show_spinner="Loading model...")
def load_model(path, mtime):  # mtime busts the cache if the file changes
    return tf.keras.models.load_model(path, compile=False)


def decode(file):
    """Decode an uploaded file to a grayscale array, or None if it is not a readable image."""
    return cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE)


def read_gray(file):
    img = decode(file)
    if img is None:
        st.error(f"Could not read **{file.name}** as an image. Please upload a valid PNG / JPG / BMP / TIF file.")
        st.stop()
    return img


def png_bytes(img_u8):
    return cv2.imencode(".png", img_u8)[1].tobytes()


def heatmap(a_u8, b_u8, gain=4):
    diff = np.clip(np.abs(a_u8.astype(np.int16) - b_u8.astype(np.int16)) * gain, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def classical_baselines(noisy_u8):
    return {"Gaussian blur": cv2.GaussianBlur(noisy_u8, (5, 5), 0),
            "Median filter": cv2.medianBlur(noisy_u8, 5),
            "Non-local means": cv2.fastNlMeansDenoising(noisy_u8, None, 15, 7, 21)}


def kv(label, value):
    return f'<div class="kv"><span>{label}</span><span>{value}</span></div>'


# ----------------------------------------------------------------------------- model + artefacts
model_path = st.sidebar.text_input("Model file", first_existing(MODEL_CANDIDATES))
full_model_path = model_path if os.path.isabs(model_path) else os.path.join(ROOT, model_path)

if not os.path.exists(full_model_path):
    st.error(f"Model file `{model_path}` not found. Run `python train.py` (or `python run_demo.py` for a "
             "quick test model) and put the resulting model file in the project folder.")
    st.stop()

model = load_model(full_model_path, os.path.getmtime(full_model_path))
SIZE = int(model.input_shape[1])
ART = os.path.dirname(full_model_path)                      # folder holding this model's results
IS_DEMO = "demo_output" in model_path
CSV = find(ART, ["denoising_metrics.csv", "metrics.csv"])
HIST_CSV = find(ART, ["training_history.csv", "history.csv"])
HIST_PNG = find(ART, ["training_history.png", "history.png"])
INFO_PATH = find(ART, ["training_info.json"])
INFO = json.load(open(INFO_PATH)) if INFO_PATH else {}
base_filters = next(l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)).filters

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="brand"><div class="logo">🫁</div><div class="name">CT Denoiser</div>'
                '<div class="sub">U-Net · Deep Learning</div></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("#### 🏗️ Architecture")
    st.markdown(
        kv("Model", "U-Net (custom)") + kv("Input", f"{SIZE}×{SIZE} grayscale") +
        kv("Encoder", f"4 blocks ({base_filters}→{base_filters * 8} ch)") +
        kv("Decoder", "4 blocks + skip conn") + kv("Loss", "Mean Squared Error") +
        kv("Optimizer", "Adam (lr=1e-3)") + kv("Parameters", f"{model.count_params():,}"),
        unsafe_allow_html=True)
    if INFO:
        st.divider()
        st.markdown("#### 🧪 Training run")
        st.markdown(
            kv("Source images", f"{INFO['source_images']:,}") +
            kv("Train / val pairs", f"{INFO['train_pairs']:,} / {INFO['val_pairs']:,}") +
            kv("Epochs run", INFO["epochs_run"]) + kv("Trained on", INFO["trained_on"]),
            unsafe_allow_html=True)
    st.divider()
    st.caption("⚠️ Research / educational demo. Not a medical device; not for diagnosis.")

# ----------------------------------------------------------------------------- hero + status
dataset_txt = INFO.get("dataset", "your dataset")
st.markdown(f"""
<div class="hero"><h3>🫁 CT Scan Denoising</h3>
<p>Upload a noisy CT / X-ray image and watch the U-Net restore it in real time.</p>
<p>Powered by TensorFlow · Evaluated with PSNR, SSIM and MSE on held-out images.</p></div>
""", unsafe_allow_html=True)
if IS_DEMO:
    st.warning("Quick **demo model trained on synthetic images**. Train on a real dataset before judging quality.")
else:
    st.success("🧠 Model loaded — ready for inference")

tab_den, tab_ana, tab_hist, tab_how = st.tabs(
    ["🫁 Denoise Image", "📊 Analytics Dashboard", "📉 Training History", "📖 How To Use"])

# ============================================================================= Denoise Image
with tab_den:
    samples = sorted(f for f in glob.glob(os.path.join(ROOT, "samples", "*"))
                     if f.lower().endswith(IMG_EXT) and not os.path.basename(f).startswith("."))
    ca, cb = st.columns(2, gap="large")
    with ca:
        st.markdown("#### 🩻 Upload Noisy Image")
        st.caption("Drag & drop a CT / X-ray image (PNG, JPG, BMP, TIF)")
        noisy_file = st.file_uploader("noisy", type=UPLOAD_TYPES, key="noisy_up", label_visibility="collapsed")
    with cb:
        st.markdown("#### 🎯 Upload Clean Reference *(optional — for metrics)*")
        st.caption("Drop the clean version for PSNR / SSIM evaluation")
        clean_file = st.file_uploader("clean", type=UPLOAD_TYPES, key="clean_up", label_visibility="collapsed")

    options = (["Sample image"] if samples else []) + ["Synthetic phantom (testing only)", "None"]
    with st.expander("No image handy? Try a sample, or simulate noise on a clean image", expanded=not noisy_file):
        e1, e2, e3 = st.columns(3)
        src = e1.radio("Clean source (noise is added to it)", options, key="src")
        noise_kind = e2.selectbox("Noise type", ["gaussian", "poisson", "saltpepper", "mixed"], key="noise")
        level = e2.slider("Noise strength", 0, 100, 50, key="level") / 100
        nseed = e3.number_input("Noise seed", 0, 9999, 1, key="nseed")
        use_clahe = e3.checkbox("CLAHE on clean image first (as in training)", True, key="clahe")
        pick = None
        if src == "Sample image":
            pick = e1.selectbox("Sample", [os.path.basename(s) for s in samples], key="sample")
        elif src.startswith("Synthetic"):
            pseed = e1.number_input("Phantom seed", 0, 9999, 7, key="pseed")

    # ---- decide what to denoise ------------------------------------------------
    clean_f, noisy_u8, note = None, None, ""
    if noisy_file is not None:                                   # real noisy image uploaded
        noisy_u8 = cv2.resize(read_gray(noisy_file), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        if clean_file is not None:
            ref = cv2.resize(read_gray(clean_file), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            clean_f = ref.astype(np.float32) / 255.0
        note = "Using your uploaded noisy image."
    else:                                                        # simulate noise on a clean image
        clean_u8 = None
        if clean_file is not None:
            clean_u8 = cv2.resize(read_gray(clean_file), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            note = "No noisy image uploaded, so noise was added to your clean reference."
        elif src == "Sample image" and pick:
            clean_u8 = cv2.resize(cv2.imread(os.path.join(ROOT, "samples", pick), cv2.IMREAD_GRAYSCALE),
                                  (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            note = "Held-out sample image with synthetic noise added."
        elif src.startswith("Synthetic"):
            clean_u8 = phantom(SIZE, np.random.default_rng(int(pseed)))
            note = "Synthetic phantom (for testing the app only)."
        if clean_u8 is not None:
            if use_clahe:
                clean_u8 = apply_clahe(clean_u8)
            clean_f = clean_u8.astype(np.float32) / 255.0
            noisy_u8 = (add_noise(clean_f, noise_kind, np.random.default_rng(int(nseed)), level=level) * 255
                        ).astype(np.uint8)

    st.divider()
    if noisy_u8 is None:
        st.info("⬆️ Upload a noisy image above, or pick a sample in the expander.")
    else:
        with st.spinner("Denoising..."):
            out_u8 = denoise(model, noisy_u8, SIZE)
        st.caption(note)
        have_ref = clean_f is not None
        cols = st.columns(4 if have_ref else 3)
        shown = ([((clean_f * 255).astype(np.uint8), "Clean reference")] if have_ref else []) + \
                [(noisy_u8, "Noisy input"), (out_u8, "Denoised (U-Net)")]
        for col, (img, cap) in zip(cols, shown):
            col.image(img, caption=cap, width="stretch", clamp=True)
        cols[-1].image(heatmap(noisy_u8, out_u8), caption="Noise removed (heatmap)", width="stretch")

        if have_ref:
            noisy_f = noisy_u8.astype(np.float32) / 255.0
            m = metrics.compare(clean_f, noisy_f, out_u8.astype(np.float32) / 255.0)
            m1, m2, m3 = st.columns(3)
            m1.metric("PSNR", f"{m['psnr_denoised']:.2f} dB", f"{m['psnr_denoised'] - m['psnr_noisy']:+.2f} dB vs noisy")
            m2.metric("SSIM", f"{m['ssim_denoised']:.3f}", f"{m['ssim_denoised'] - m['ssim_noisy']:+.3f} vs noisy")
            m3.metric("MSE reduction", f"{100 * (1 - m['mse_denoised'] / m['mse_noisy']):.1f}%")

            if st.checkbox("Compare with classical filters", True, key="classical"):
                results = {"Noisy (no denoising)": noisy_u8, **classical_baselines(noisy_u8), "U-Net": out_u8}
                table = pd.DataFrame(
                    {k: {"PSNR (dB)": metrics.psnr(clean_f, v.astype(np.float32) / 255.0),
                         "SSIM": metrics.ssim(clean_f, v.astype(np.float32) / 255.0)}
                     for k, v in results.items()}).T.round(3)
                t1, t2 = st.columns(2)
                t1.dataframe(table, width="stretch")
                t2.bar_chart(table["PSNR (dB)"])
        else:
            st.info("Add a clean reference image to see PSNR / SSIM for this result.")

        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("⬇️ Denoised image", png_bytes(out_u8), "denoised.png", "image/png")
        d2.download_button("⬇️ Side-by-side", png_bytes(np.hstack([noisy_u8, out_u8])), "comparison.png", "image/png")

    with st.expander("📦 Batch mode — denoise many images and download a ZIP"):
        files = st.file_uploader("Upload images", type=UPLOAD_TYPES, accept_multiple_files=True, key="batch")
        if files:
            buf, previews, bar = io.BytesIO(), [], st.progress(0.0)
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for i, f in enumerate(files):
                    img = decode(f)
                    if img is None:
                        st.warning(f"Skipped {f.name}: not a readable image.")
                        continue
                    out = denoise(model, img, SIZE)
                    z.writestr(os.path.splitext(f.name)[0] + "_denoised.png", png_bytes(out))
                    if len(previews) < 3:
                        previews.append((img, out, f.name))
                    bar.progress((i + 1) / len(files))
            st.success(f"Denoised {len(files)} image(s).")
            st.download_button("⬇️ Download ZIP", buf.getvalue(), "denoised_images.zip", "application/zip")
            for img, out, name in previews:
                a, b = st.columns(2)
                a.image(img, caption=f"{name} (input)", width="stretch")
                b.image(out, caption="denoised", width="stretch")

# ============================================================================= Analytics Dashboard
with tab_ana:
    if CSV is None:
        st.info("No metrics file yet. Run `python train.py` — it evaluates the model on held-out images and writes "
                "`denoising_metrics.csv`; the dashboard fills in automatically.")
    else:
        df = pd.read_csv(CSV)
        st.subheader(f"Held-out evaluation · {len(df)} images the model never trained on")
        if INFO:
            st.markdown(f'<span class="badge">{INFO["dataset"]}</span>'
                        f'<span class="badge">{INFO["img_size"]}×{INFO["img_size"]}</span>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PSNR", f"{df.psnr_denoised.mean():.2f} dB", f"{df.psnr_denoised.mean() - df.psnr_noisy.mean():+.2f} dB")
        c2.metric("SSIM", f"{df.ssim_denoised.mean():.3f}", f"{df.ssim_denoised.mean() - df.ssim_noisy.mean():+.3f}")
        c3.metric("MSE reduction", f"{100 * (1 - df.mse_denoised.mean() / df.mse_noisy.mean()):.1f}%")
        c4.metric("Images improved (PSNR)", f"{100 * (df.psnr_denoised > df.psnr_noisy).mean():.0f}%")

        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        for a, col, title in zip(ax[:2], ["psnr", "ssim"], ["PSNR (dB)", "SSIM"]):
            a.hist(df[f"{col}_noisy"], bins=20, alpha=.6, label="noisy")
            a.hist(df[f"{col}_denoised"], bins=20, alpha=.6, label="denoised")
            a.set_title(title); a.legend()
        ax[2].scatter(df.psnr_noisy, df.psnr_denoised, s=12)
        lo = min(df.psnr_noisy.min(), df.psnr_denoised.min()); hi = max(df.psnr_noisy.max(), df.psnr_denoised.max())
        ax[2].plot([lo, hi], [lo, hi], "r--"); ax[2].set_title("Per-image PSNR gain")
        ax[2].set_xlabel("noisy"); ax[2].set_ylabel("denoised")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        with st.expander("Per-image metrics"):
            st.dataframe(df.round(4), width="stretch")
        st.download_button("⬇️ Download metrics CSV", df.to_csv(index=False), "denoising_metrics.csv", "text/csv")

# ============================================================================= Training History
with tab_hist:
    if HIST_CSV is None and HIST_PNG is None:
        st.info("No training history yet. `python train.py` saves `training_history.csv/.png` here.")
    else:
        if HIST_CSV:
            h = pd.read_csv(HIST_CSV)
            best = int(h.val_loss.idxmin())
            k1, k2, k3 = st.columns(3)
            k1.metric("Epochs run", len(h))
            k2.metric("Best val loss (MSE)", f"{h.val_loss.min():.5f}", f"epoch {best + 1}")
            k3.metric("Final train loss", f"{h.loss.iloc[-1]:.5f}")
            st.line_chart(h[["loss", "val_loss"]], x_label="epoch", y_label="MSE loss")
        elif HIST_PNG:
            st.image(HIST_PNG, width="stretch")

# ============================================================================= How To Use
with tab_how:
    st.markdown(f"""
### How to use this app
1. Go to **Denoise Image** and upload a noisy CT / X-ray image (or use the sample in the expander).
2. *(Optional)* upload the **clean reference** version to get PSNR / SSIM / MSE.
3. Don't have a noisy image? Pick a clean one and the app **adds synthetic noise** (Gaussian, Poisson, salt-and-pepper)
   with a strength slider, then denoises it.
4. Compare the **noisy input**, the **U-Net result** and a **heatmap of what was removed**, and see how the U-Net
   stacks up against classical filters.
5. Download the denoised image, or use **Batch mode** for many images at once.

### What happens inside
grayscale → resize to {SIZE}×{SIZE} → U-Net (encoder → bottleneck → decoder with skip connections) → denoised image.
The network was trained on pairs of *noisy → clean* images to minimise mean squared error.

### Reading the metrics
| Metric | Meaning |
|---|---|
| **PSNR** (dB) | Higher = closer to the clean image |
| **SSIM** | 1.0 = identical structure |
| **MSE reduction** | How much the pixel error fell vs the noisy input |

**Notes:** inputs are resized to {SIZE}×{SIZE}. Training used *synthetic* noise, so results on a scanner's real noise
may differ. Research / educational demo only — not for diagnosis.
""")

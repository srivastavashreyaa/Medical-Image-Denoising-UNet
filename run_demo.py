"""One-command sanity check: runs the WHOLE pipeline on synthetic phantom images (small + fast).

    python run_demo.py

Everything is written to ./demo_output. The images are synthetic, so the numbers are only a
check that your setup works - they are NOT results to report. Use a real dataset for that.
"""
import os
import subprocess
import sys

OUT = "demo_output"
PY = sys.executable  # same interpreter / venv that launched this script (works on Windows too)


def run(*args):
    print("\n$", "python", *args, flush=True)
    r = subprocess.run([PY, *args])
    if r.returncode != 0:
        sys.exit(f"Step failed: {' '.join(args)}")


os.makedirs(OUT, exist_ok=True)
raw, data = os.path.join(OUT, "raw"), os.path.join(OUT, "data")
run("generate_demo_data.py", "--output_dir", raw, "--count", "40", "--img_size", "64")
run("preprocess.py", "--input_dir", raw, "--output_dir", data, "--img_size", "64", "--augment", "2")
run("train.py", "--data_dir", data, "--img_size", "64", "--epochs", "10", "--batch_size", "8",
    "--base_filters", "8", "--model_out", os.path.join(OUT, "model.keras"),
    "--metrics_out", os.path.join(OUT, "metrics.csv"),
    "--history_out", os.path.join(OUT, "history.png"))
run("visualize.py", "--metrics", os.path.join(OUT, "metrics.csv"), "--output_dir", OUT)
run("compare.py", "--data_dir", data, "--model", os.path.join(OUT, "model.keras"),
    "--img_size", "64", "--n", "3", "--output", os.path.join(OUT, "comparison.png"))
print(f"\nDEMO OK - open the images in ./{OUT}/ (comparison.png, history.png, ...).")

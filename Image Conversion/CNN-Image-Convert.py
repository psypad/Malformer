import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed   # for parallel processing

# ---------- CONFIG ----------
BYTES_DIR = "dataset/train"          # where .bytes files are
LABELS_FILE = "dataset/trainLabels.csv"
OUTPUT_DIR = "images128"             # output folder for images
IMG_SIZE = (128,128)
N_JOBS = 8                           # parallel workers

# ---------- FUNCTIONS ----------
def read_bytes_file(filepath, max_len=16384):
    """Read first max_len bytes from .bytes file (ignoring addresses)."""
    arr = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            # Skip first token (hex address)
            for token in parts[1:]:
                if token != "??":  # ?? means unknown byte
                    arr.append(int(token, 16))
                else:
                    arr.append(0)
                if len(arr) >= max_len:
                    return np.array(arr, dtype=np.uint8)
    return np.array(arr + [0]*(max_len-len(arr)), dtype=np.uint8)

def bytes_to_image(input_file, output_file, size=IMG_SIZE):
    """Convert .bytes → grayscale image and save."""
    data = read_bytes_file(input_file, size[0]*size[1])
    arr = data.reshape(size)
    img = Image.fromarray(arr, mode="L")
    img.save(output_file)

def process_file(row, out_base):
    file_id, label = row["Id"], row["Class"]
    in_file = Path(BYTES_DIR) / f"{file_id}.bytes"
    out_dir = Path(out_base) / f"class{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{file_id}.png"
    try:
        bytes_to_image(in_file, out_file)
    except Exception as e:
        print(f"Error with {file_id}: {e}")

# ---------- MAIN ----------
def main():
    labels = pd.read_csv(LABELS_FILE)
    Parallel(n_jobs=N_JOBS)(
        delayed(process_file)(row, OUTPUT_DIR)
        for _, row in tqdm(labels.iterrows(), total=len(labels))
    )
    print("✅ Conversion complete.")

if __name__ == "__main__":
    main()

import os
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import csv

# ---------------- FIXED PATHS ----------------
BYTES_DIR = Path("E:/capstone/test")           # where your .bytes files are
LABELS_FILE = Path("E:/capstone/trainLabels.csv")
OUTPUT_DIR = Path("E:/capstone/ImageDatasets/CNN128")    # output images go here
IMG_SIZE = (128, 128)                          # width, height
METHOD = "colormap_truncate"                   # "grayscale_truncate" | "colormap_truncate" | "3gram"

# ---------------- HELPERS ----------------
def read_bytes_file(filepath, max_len):
    """Read textual .bytes file, keep first max_len bytes, convert to numpy array."""
    arr = []
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= 1:
                continue
            for token in parts[1:]:
                if token == "??":
                    arr.append(0)
                else:
                    try:
                        arr.append(int(token, 16))
                    except:
                        arr.append(0)
                if len(arr) >= max_len:
                    return np.array(arr, dtype=np.uint8)
    # pad if too short
    if len(arr) < max_len:
        arr.extend([0] * (max_len - len(arr)))
    return np.array(arr, dtype=np.uint8)


def make_colormap_palette(name="plasma"):
    """Generate a 256-color palette (requires matplotlib, else fallback grayscale)."""
    try:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(name, 256)
        return (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)
    except:
        return np.stack([np.arange(256)] * 3, axis=1).astype(np.uint8)


def bytes_to_image(file_id, label, method=METHOD, img_size=IMG_SIZE, cmap_rgb=None):
    """Convert one .bytes file to an image and save in class folder."""
    in_file = BYTES_DIR / f"{file_id}.bytes"
    if not in_file.exists():
        return None

    w, h = img_size
    if method == "3gram":
        max_len = w * h * 3
    else:
        max_len = w * h

    bts = read_bytes_file(in_file, max_len)

    if method == "grayscale_truncate":
        arr = bts.reshape((h, w))
        img = Image.fromarray(arr, mode="L")

    elif method == "colormap_truncate":
        arr = bts.reshape((h, w))
        rgb = cmap_rgb[arr]
        img = Image.fromarray(rgb, mode="RGB")

    elif method == "3gram":
        arr = bts.reshape((h, w, 3))
        arr[:, :, 2] = 255 - arr[:, :, 2]  # tweak B channel
        img = Image.fromarray(arr, mode="RGB")

    else:
        raise ValueError("Unknown method")

    out_dir = OUTPUT_DIR / method / f"class{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{file_id}.png"
    img.save(out_path)
    return out_path


# ---------------- MAIN ----------------
def main():
    labels = pd.read_csv(LABELS_FILE)

    # only keep rows where .bytes file exists in E:/capstone/test
    labels = labels[labels["Id"].apply(lambda x: (BYTES_DIR / f"{x}.bytes").exists())]
    labels = labels.reset_index(drop=True)

    cmap_rgb = make_colormap_palette("plasma") if METHOD == "colormap_truncate" else None

    created = []
    for _, row in tqdm(labels.iterrows(), total=len(labels), desc="Converting"):
        fid, lab = row["Id"], row["Class"]
        outp = bytes_to_image(fid, lab, METHOD, IMG_SIZE, cmap_rgb)
        if outp:
            created.append((str(outp), lab))

    # write label CSV
    out_csv = OUTPUT_DIR / f"labels_{METHOD}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "class"])
        for path, lab in created:
            w.writerow([path, lab])

    print(f"✅ Done. {len(created)} images created.")
    print(f"Labels saved to {out_csv}")


if __name__ == "__main__":
    main()

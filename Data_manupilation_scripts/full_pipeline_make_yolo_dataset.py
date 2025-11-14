#!/usr/bin/env python3
"""
Full pipeline:
  - Read dataset root with class subfolders
  - Generate labels CSV (relative paths)
  - Create stratified 80/10/10 split into dst/train/<class>, dst/val/<class>, dst/test/<class>
  - Write YOLO-style data.yaml to dst
  - Optionally write train.txt/val.txt/test.txt with absolute paths (one per line)

Usage example:
  python full_pipeline_make_yolo_dataset.py \
    --src /path/to/colormap_truncate \
    --dst /path/to/output_dataset \
    --csv labels_generated.csv \
    --seed 123 \
    --symlink   # optional: create symlinks instead of copying
    --make-lists  # optional: create train.txt/val.txt/test.txt with absolute paths
"""

import argparse
from pathlib import Path
import random
import shutil
import math
import csv
import yaml
from typing import List

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def list_images(folder: Path) -> List[Path]:
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def split_counts(n:int, train_frac:float, val_frac:float, test_frac:float):
    n_train = int(math.floor(n * train_frac))
    n_val   = int(math.floor(n * val_frac))
    n_test  = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = max(0, n - n_train)
    return n_train, n_val, n_test

def create_splits(files: List[Path], train_frac, val_frac, test_frac, rng):
    n = len(files)
    n_train, n_val, n_test = split_counts(n, train_frac, val_frac, test_frac)
    shuffled = files.copy()
    rng.shuffle(shuffled)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:n_train + n_val + n_test]
    assert len(train) + len(val) + len(test) == n
    return train, val, test

def copy_or_symlink(files: List[Path], dest_folder: Path, symlink=False):
    safe_mkdir(dest_folder)
    out_paths = []
    for src in files:
        dest = dest_folder / src.name
        # ensure unique filename in dest; if conflict, append index
        if dest.exists():
            base = dest.stem
            suff = dest.suffix
            i = 1
            while (dest_folder / f"{base}__{i}{suff}").exists():
                i += 1
            dest = dest_folder / f"{base}__{i}{suff}"
        try:
            if symlink:
                # relative symlink if possible
                rel = src.resolve().relative_to(dest_folder.resolve().parent) if False else src.resolve()
                dest.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dest)
        except Exception:
            # fallback to copy if symlink fails
            shutil.copy2(src, dest)
        out_paths.append(dest)
    return out_paths

def write_labels_csv(rows, out_csv: Path):
    with out_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['relative_path','class_name','class_id'])
        writer.writerows(rows)

def write_data_yaml(dst_root: Path, classes: List[str], yaml_path: Path):
    data = {
        'train': str((dst_root / 'train').resolve()),
        'val':   str((dst_root / 'val').resolve()),
        'test':  str((dst_root / 'test').resolve()),
        'nc':    len(classes),
        'names': classes
    }
    with yaml_path.open('w') as f:
        yaml.dump(data, f, sort_keys=False)
    return data

def write_list_file(list_paths: List[Path], out_file: Path):
    with out_file.open('w') as f:
        for p in list_paths:
            f.write(str(p.resolve()) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Create labels CSV + 80/10/10 stratified splits + YOLO data.yaml")
    parser.add_argument("--src", required=True, type=Path, help="Source root with class subfolders")
    parser.add_argument("--dst", required=True, type=Path, help="Destination root to create train/val/test")
    parser.add_argument("--csv", default="labels_generated.csv", type=Path, help="Output labels CSV (relative paths)")
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symlink", action='store_true', help="Symlink files into the new structure instead of copying")
    parser.add_argument("--min-per-class", type=int, default=1, help="Warn if class has fewer images")
    parser.add_argument("--make-lists", action='store_true', help="Also produce train.txt/val.txt/test.txt with absolute paths")
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    rng = random.Random(args.seed)

    if not src.exists():
        raise SystemExit(f"Source folder does not exist: {src}")
    # find class folders
    classes = [d for d in sorted(src.iterdir()) if d.is_dir()]
    if not classes:
        raise SystemExit(f"No class subfolders found in: {src}")

    print(f"Found {len(classes)} classes.")
    class_names = [c.name for c in classes]
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    print("Class -> id mapping:", class_to_id)

    # prepare dst folders
    train_root = dst / 'train'
    val_root = dst / 'val'
    test_root = dst / 'test'

    labels_rows = []  # rows for csv: (relative_path, class_name, class_id)
    train_list_abs = []
    val_list_abs = []
    test_list_abs = []

    total_counts = {'total':0, 'train':0, 'val':0, 'test':0}

    for cls in classes:
        imgs = list_images(cls)
        n = len(imgs)
        if n < args.min_per_class:
            print(f"WARNING: class {cls.name} has only {n} images (min-per-class={args.min_per_class})")
        train_files, val_files, test_files = create_splits(imgs, args.train_frac, args.val_frac, args.test_frac, rng)

        # copy or symlink into dest
        out_train = copy_or_symlink(train_files, train_root / cls.name, symlink=args.symlink)
        out_val   = copy_or_symlink(val_files,   val_root / cls.name,   symlink=args.symlink)
        out_test  = copy_or_symlink(test_files,  test_root / cls.name,  symlink=args.symlink)

        # Add CSV rows; relative paths relative to dst root
        for p in out_train:
            rel = p.relative_to(dst)
            labels_rows.append([str(rel.as_posix()), cls.name, class_to_id[cls.name]])
            if args.make_lists:
                train_list_abs.append(p)
        for p in out_val:
            rel = p.relative_to(dst)
            labels_rows.append([str(rel.as_posix()), cls.name, class_to_id[cls.name]])
            if args.make_lists:
                val_list_abs.append(p)
        for p in out_test:
            rel = p.relative_to(dst)
            labels_rows.append([str(rel.as_posix()), cls.name, class_to_id[cls.name]])
            if args.make_lists:
                test_list_abs.append(p)

        total_counts['total'] += n
        total_counts['train'] += len(out_train)
        total_counts['val'] += len(out_val)
        total_counts['test'] += len(out_test)

        print(f"{cls.name}: total={n}, train={len(out_train)}, val={len(out_val)}, test={len(out_test)}")

    # write labels csv
    csv_out = args.csv if args.csv.is_absolute() else (dst / args.csv)
    safe_mkdir(csv_out.parent)
    write_labels_csv(labels_rows, csv_out)
    print(f"Wrote labels CSV: {csv_out}")

    # write data.yaml
    yaml_path = dst / 'data.yaml'
    data_yaml = write_data_yaml(dst, class_names, yaml_path)
    print(f"Wrote YOLO data.yaml: {yaml_path}")

    # optionally write train/val/test list files (absolute paths)
    if args.make_lists:
        write_list_file(train_list_abs, dst / 'train.txt')
        write_list_file(val_list_abs,   dst / 'val.txt')
        write_list_file(test_list_abs,  dst / 'test.txt')
        print(f"Wrote train/val/test lists to {dst}/*.txt")

    # final summary
    print("\nTotals:")
    print(f"  images total: {total_counts['total']}")
    print(f"  train: {total_counts['train']}")
    print(f"  val:   {total_counts['val']}")
    print(f"  test:  {total_counts['test']}")
    print("\nPipeline complete. You can now point YOLO to:")
    print(f"  train: {data_yaml['train']}")
    print(f"  val:   {data_yaml['val']}")
    print(f"  test:  {data_yaml['test']}")
    print(f"  classes (nc): {data_yaml['nc']}")
    print("Note: data.yaml 'names' ordering matches class folder sort order.")

if __name__ == "__main__":
    main()

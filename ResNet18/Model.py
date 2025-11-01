# Model.py (robust directory handling)
import argparse
import os
from pathlib import Path
import random
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k * 100.0 / target.size(0)).item())
        return res[0] if len(res)==1 else res

def compute_class_weights_from_folder(folder):
    ds = ImageFolder(folder)
    counts = {}
    for _, lab in ds.samples:
        counts[lab] = counts.get(lab,0) + 1
    max_label = max(counts.keys())
    freq = np.array([counts.get(i, 0) for i in range(max_label+1)], dtype=np.float32)
    freq = np.where(freq==0, 1.0, freq)
    weights = 1.0 / freq
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)

def build_loaders(data_dir, img_size=(128,128), batch_size=64, val_split=0.15, workers=4):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size[0]*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    full = ImageFolder(data_dir, transform=train_tf)
    num_classes = len(full.classes)
    n_total = len(full)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_set, val_set = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    val_set.dataset.transform = val_tf

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, num_classes, full.class_to_idx

def build_model(num_classes, pretrained=True):
    model = resnet18(pretrained=pretrained)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)
    return model

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    for x,y in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        bs = x.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(out, y) * bs / 100.0
        n += bs
    return running_loss / n, (running_acc / n) * 100.0

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    with torch.no_grad():
        for x,y in tqdm(loader, leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            bs = x.size(0)
            running_loss += loss.item() * bs
            running_acc += accuracy(out, y) * bs / 100.0
            n += bs
    return running_loss / n, (running_acc / n) * 100.0

def safe_makedirs(path_str):
    """
    Create directories robustly. If path is on an unavailable drive, raise an informative error.
    """
    try:
        os.makedirs(path_str, exist_ok=True)
        return True
    except Exception as e:
        return False

def resolve_output_dir(preferred):
    """
    Ensure preferred output dir is usable. If not, fallback to local ./models/resnet18_colormap
    """
    pref = Path(preferred)
    # try to create parent directories
    ok = safe_makedirs(str(pref))
    if ok:
        return pref
    # fallback to project-local models folder
    fallback = Path.cwd() / "models" / "resnet18_colormap"
    safe_makedirs(str(fallback))
    print(f"Warning: couldn't create preferred output dir '{preferred}'. Falling back to '{fallback}'.")
    return fallback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="Data/colormap_truncate", help="path to ImageFolder-style data")
    parser.add_argument("--output_dir", default="E:/capstone/models/resnet18_colormap", help="preferred where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = resolve_output_dir(args.output_dir)
    print(f"Using output directory: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, num_classes, class_to_idx = build_loaders(
        args.data_dir, img_size=(128,128), batch_size=args.batch_size, val_split=0.15, workers=args.workers
    )
    print(f"Found classes: {num_classes}, mapping sample: {dict(list(class_to_idx.items())[:5])}")

    model = build_model(num_classes, pretrained=args.pretrained).to(device)

    # class weights
    weights = compute_class_weights_from_folder(args.data_dir).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    scaler = torch.cuda.amp.GradScaler() if (args.use_amp and torch.cuda.is_available()) else None

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}/{args.epochs}")
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        print(f" Train Loss: {t_loss:.4f} Train Acc: {t_acc:.2f}%")
        v_loss, v_acc = validate(model, val_loader, criterion, device)
        print(f" Val   Loss: {v_loss:.4f} Val   Acc: {v_acc:.2f}%")

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc": v_acc,
            "class_to_idx": class_to_idx
        }
        # save last
        try:
            torch.save(ckpt, out_dir / "last.pth")
        except Exception as e:
            print(f"Warning: failed to save last.pth to {out_dir}: {e}")

        # save best
        if v_acc > best_acc:
            best_acc = v_acc
            try:
                torch.save(ckpt, out_dir / "best.pth")
                print(" Saved best.pth")
            except Exception as e:
                print(f"Warning: failed to save best.pth to {out_dir}: {e}")

        scheduler.step()

    try:
        torch.save({"model_state": model.state_dict(), "class_to_idx": class_to_idx}, out_dir / "final.pth")
    except Exception as e:
        print(f"Warning: failed to save final.pth to {out_dir}: {e}")

    print("Training complete. Best val acc: {:.2f}%".format(best_acc))

if __name__ == "__main__":
    main()

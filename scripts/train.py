"""
Phase 2, Step 2: train BiLSTMClassifier, LSTMAttentionClassifier, and
TransformerCrashClassifier on the tensors produced by build_sequences.py.

Usage:
    .venv\\Scripts\\python.exe scripts\\train.py                # trains all 3
    .venv\\Scripts\\python.exe scripts\\train.py --model transformer
    .venv\\Scripts\\python.exe scripts\\train.py --epochs 80 --patience 10

Loss: per-sample-weighted cross-entropy, weight = class_weight x
source_weight (computed once in build_sequences.py and saved as
sample_weight_train.npy). This is the README's "Final Sample Weight =
Source Weight x Class Weight" bias-mitigation formula --
nn.CrossEntropyLoss's built-in `weight=` argument only supports a
per-class weight, not a per-sample one, so here the loss is computed with
reduction="none" and manually weighted-averaged instead.

Model selection: best checkpoint = highest macro-F1 on the validation
split (SynSHRP2+BeamNG held-out 15%, never CISS). Macro-F1 (not accuracy)
because the class distribution is imbalanced (Rear-end is ~4x
Sideswipe in the train split) -- accuracy alone would reward a model that
just predicts the majority class often. Early stopping (patience, default
8 epochs without a val macro-F1 improvement) prevents wasting GPU time and
overfitting once the model has plateaued.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import NUM_CLASSES, RANDOM_STATE, model_dir, tensor_dir  # noqa: E402
from models import build_model  # noqa: E402

MODEL_NAMES = {
    "bilstm": "BiLSTMClassifier",
    "lstm_attn": "LSTMAttentionClassifier",
    "transformer": "TransformerCrashClassifier",
}

# Set by main() from --split-mode. Checkpoints (and the training-curve
# history.json alongside them) for a given split live under
# models/<split>/; results/<split>/ is reserved for evaluate.py's
# test-set outputs (confusion matrices, prediction arrays, summaries).
_TENSOR_DIR = tensor_dir("source")
_MODEL_DIR = model_dir("source")


def load_split(split):
    X = np.load(_TENSOR_DIR / f"X_{split}.npy")
    y = np.load(_TENSOR_DIR / f"y_{split}.npy")
    mask = np.load(_TENSOR_DIR / f"mask_{split}.npy")
    return X, y, mask


def make_loader(X, y, mask, weight=None, batch_size=64, shuffle=True):
    tensors = [torch.from_numpy(X).float(), torch.from_numpy(y).long(),
               torch.from_numpy(mask).bool()]
    if weight is not None:
        tensors.append(torch.from_numpy(weight).float())
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss, all_preds, all_true = 0.0, [], []
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for batch in loader:
            if train_mode:
                X_b, y_b, mask_b, w_b = batch
                w_b = w_b.to(device)
            else:
                X_b, y_b, mask_b = batch
                w_b = None
            X_b, y_b, mask_b = X_b.to(device), y_b.to(device), mask_b.to(device)

            if train_mode:
                optimizer.zero_grad()
            out = model(X_b, pad_mask=mask_b)
            logits = out[0] if isinstance(out, tuple) else out

            per_sample_loss = nn.functional.cross_entropy(logits, y_b, reduction="none")
            loss = (per_sample_loss * w_b).mean() if w_b is not None else per_sample_loss.mean()

            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * len(y_b)
            all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
            all_true.extend(y_b.cpu().numpy())

    n = len(all_true)
    macro_f1 = f1_score(all_true, all_preds, average="macro", labels=list(range(NUM_CLASSES)))
    return total_loss / n, macro_f1


def train_one(model_key, epochs, batch_size, lr, patience, device):
    class_name = MODEL_NAMES[model_key]
    print(f"\n{'=' * 70}\nTraining {class_name} ({model_key})\n{'=' * 70}")

    X_train, y_train, mask_train = load_split("train")
    sample_w = np.load(_TENSOR_DIR / "sample_weight_train.npy")
    X_val, y_val, mask_val = load_split("val")

    train_loader = make_loader(X_train, y_train, mask_train, sample_w,
                                batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, mask_val, batch_size=batch_size, shuffle=False)

    torch.manual_seed(RANDOM_STATE)
    model = build_model(model_key, input_dim=X_train.shape[-1], num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} | device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_f1, patience_ctr = -1.0, 0
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "lr": []}
    ckpt_path = _MODEL_DIR / f"{class_name}_best.pt"

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_f1 = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_f1 = run_epoch(model, val_loader, device, optimizer=None)
        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_f1)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        marker = ""
        if val_f1 > best_val_f1:
            best_val_f1, patience_ctr = val_f1, 0
            torch.save(model.state_dict(), ckpt_path)
            marker = "  <- saved"
        else:
            patience_ctr += 1

        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} (F1 {train_f1:.4f}) | "
              f"val_loss {val_loss:.4f} | val_macro_F1 {val_f1:.4f}{marker}")

        if patience_ctr >= patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
            break

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s. Best val macro-F1 = {best_val_f1:.4f}. Checkpoint: {ckpt_path}")

    history["best_val_macro_f1"] = best_val_f1
    history["n_params"] = n_params
    history["train_seconds"] = elapsed
    history["epochs_run"] = len(history["train_loss"])
    with open(_MODEL_DIR / f"{class_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return best_val_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_NAMES) + ["all"], default="all")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--split-mode", choices=["source", "mixed"], default="source",
                         help="which build_sequences.py output to train on: 'source' "
                              "(default, strict split) reads models/strict/, 'mixed' reads "
                              "models/mixed/. Checkpoints and training curves are written "
                              "back into that same directory.")
    args = parser.parse_args()

    global _TENSOR_DIR, _MODEL_DIR
    _TENSOR_DIR = tensor_dir(args.split_mode)
    _MODEL_DIR = model_dir(args.split_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    keys = list(MODEL_NAMES) if args.model == "all" else [args.model]
    results = {}
    for key in keys:
        results[key] = train_one(key, args.epochs, args.batch_size, args.lr, args.patience, device)

    print(f"\n{'=' * 70}\nSummary (best validation macro-F1)\n{'=' * 70}")
    for key, f1 in results.items():
        print(f"  {MODEL_NAMES[key]:<28} {f1:.4f}")


if __name__ == "__main__":
    main()

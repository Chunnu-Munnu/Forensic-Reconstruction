"""
Secondary diagnostic (documentation.txt Part 9.4): trains and evaluates
TransformerCrashClassifier on a stratified RANDOM split that pools all
three sources together, instead of the primary source-based split. This
answers a narrower question than the main Phase 2 result -- "do the UIR
features and this architecture support 5-class crash classification at
all when train and test come from the same distribution?" -- isolated
from the much harder question the main result answers ("does a model
trained on synthetic/naturalistic data generalize to real CISS crashes?").

Run build_sequences.py --split-mode random first:
    .venv\\Scripts\\python.exe scripts\\build_sequences.py --split-mode random
    .venv\\Scripts\\python.exe scripts\\random_split_diagnostic.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_NAMES, NUM_CLASSES, RANDOM_STATE, model_dir, results_dir, tensor_dir  # noqa: E402
from models import build_model  # noqa: E402
from train import make_loader, run_epoch  # noqa: E402

TENSOR_DIR = tensor_dir("random")
OUT_MODEL_DIR = model_dir("random")
OUT_RESULTS_DIR = results_dir("random")


def load(split):
    X = np.load(TENSOR_DIR / f"X_{split}.npy")
    y = np.load(TENSOR_DIR / f"y_{split}.npy")
    mask = np.load(TENSOR_DIR / f"mask_{split}.npy")
    return X, y, mask


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, mask_train = load("train")
    sample_w = np.load(TENSOR_DIR / "sample_weight_train.npy")
    X_val, y_val, mask_val = load("val")
    X_test, y_test, mask_test = load("test")

    print(f"train={len(y_train)} val={len(y_val)} test={len(y_test)}")

    train_loader = make_loader(X_train, y_train, mask_train, sample_w, batch_size=64, shuffle=True)
    val_loader = make_loader(X_val, y_val, mask_val, batch_size=64, shuffle=False)

    torch.manual_seed(RANDOM_STATE)
    model = build_model("transformer", input_dim=X_train.shape[-1], num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_f1, patience_ctr, best_state = -1.0, 0, None
    for epoch in range(1, 61):
        train_loss, train_f1 = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_f1 = run_epoch(model, val_loader, device, optimizer=None)
        scheduler.step(val_f1)
        marker = ""
        if val_f1 > best_f1:
            best_f1, patience_ctr = val_f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = "  <- saved"
        else:
            patience_ctr += 1
        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} (F1 {train_f1:.4f}) | "
              f"val_loss {val_loss:.4f} | val_macro_F1 {val_f1:.4f}{marker}")
        if patience_ctr >= 8:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, OUT_MODEL_DIR / "TransformerCrashClassifier_best.pt")

    model.eval()
    preds = []
    with torch.no_grad():
        X_t = torch.from_numpy(X_test).float().to(device)
        mask_t = torch.from_numpy(mask_test).bool().to(device)
        for i in range(0, len(X_t), 256):
            logits = model(X_t[i:i + 256], pad_mask=mask_t[i:i + 256])
            preds.append(logits.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(preds)

    print(f"\n{'=' * 70}\nTransformerCrashClassifier -- RANDOM split test set (n={len(y_test)})\n{'=' * 70}")
    print("Class support in this test split:",
          {CLASS_NAMES[c]: int((y_test == c).sum()) for c in range(NUM_CLASSES)})
    report = classification_report(y_test, preds, labels=list(range(NUM_CLASSES)),
                                    target_names=CLASS_NAMES, zero_division=0)
    print(report)
    macro_f1 = f1_score(y_test, preds, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)
    acc = (preds == y_test).mean()
    print(f"Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}")

    OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_RESULTS_DIR / "random_split_diagnostic_report.txt"
    with open(report_path, "w") as f:
        f.write(f"train={len(y_train)} val={len(y_val)} test={len(y_test)}\n")
        f.write(f"best_val_macro_f1={best_f1:.4f}\n")
        f.write(f"test_accuracy={acc:.4f}\n")
        f.write(f"test_macro_f1={macro_f1:.4f}\n\n")
        f.write(report)
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()

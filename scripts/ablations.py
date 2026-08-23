"""
Phase 2, Step 4: two ablation studies on TransformerCrashClassifier (the
primary model), both evaluated on the same CISS test split as the main
result for a fair comparison. See documentation.txt Part 8 for full
rationale; short version:

Ablation 1 -- drop the four *_missing indicator columns, train on the raw
7 signal columns only. Tests whether explicitly telling the model "this
value is a placeholder, not a real zero" (this project's core data
philosophy, per README/CLAUDE.md) actually earns its keep.

Ablation 2 -- force steering_deg=0 and steering_deg_missing=1 for every
row (as if no vehicle in the dataset had a steering sensor). Quantifies
how much the model leans on a signal that's genuinely present for only a
small fraction of events in the first place.

Usage:
    .venv\\Scripts\\python.exe scripts\\ablations.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sequences import build_sequences, load_and_clean  # noqa: E402
from config import (  # noqa: E402
    CLASS_NAMES, NUM_CLASSES, RANDOM_STATE, RAW_FEATURE_COLS,
    SEQ_LEN, SOURCE_WEIGHTS, TEST_SOURCE, TRAIN_SOURCES, results_dir,
)
from models import build_model  # noqa: E402
from train import make_loader, run_epoch  # noqa: E402

# Ablation 1's feature set: the 7 raw signals, no *_missing indicators.
NO_INDICATOR_FEATURES = list(RAW_FEATURE_COLS)


def split_and_scale(X, y, pad_mask, sources):
    is_train_pool = np.isin(sources, TRAIN_SOURCES)
    is_test = sources == TEST_SOURCE
    train_pool_idx = np.where(is_train_pool)[0]
    test_idx = np.where(is_test)[0]

    train_idx, val_idx = train_test_split(
        train_pool_idx, test_size=0.15, random_state=RANDOM_STATE, stratify=y[train_pool_idx])

    N_tr, T_, F_ = X[train_idx].shape
    scaler = StandardScaler().fit(X[train_idx].reshape(-1, F_))

    def scale(idx):
        n = len(idx)
        return scaler.transform(X[idx].reshape(-1, F_)).reshape(n, T_, F_).astype(np.float32)

    class_w = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y[train_idx])
    class_w_map = {c: w for c, w in enumerate(class_w)}
    sample_w = np.array([class_w_map[label] * SOURCE_WEIGHTS.get(src, 1.0)
                          for label, src in zip(y[train_idx], sources[train_idx])], dtype=np.float32)

    return (scale(train_idx), y[train_idx], pad_mask[train_idx], sample_w,
            scale(val_idx), y[val_idx], pad_mask[val_idx],
            scale(test_idx), y[test_idx], pad_mask[test_idx])


def train_and_eval(name, X_train, y_train, mask_train, sample_w,
                    X_val, y_val, mask_val, X_test, y_test, mask_test, device):
    print(f"\n{'=' * 70}\nAblation: {name}\n{'=' * 70}")
    train_loader = make_loader(X_train, y_train, mask_train, sample_w, batch_size=64, shuffle=True)
    val_loader = make_loader(X_val, y_val, mask_val, batch_size=64, shuffle=False)

    torch.manual_seed(RANDOM_STATE)
    model = build_model("transformer", input_dim=X_train.shape[-1], num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_f1, patience_ctr, best_state = -1.0, 0, None
    for epoch in range(1, 61):
        _, _ = run_epoch(model, train_loader, device, optimizer)
        _, val_f1 = run_epoch(model, val_loader, device, optimizer=None)
        scheduler.step(val_f1)
        if val_f1 > best_f1:
            best_f1, patience_ctr = val_f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
        if patience_ctr >= 8:
            break
    print(f"  best val macro-F1: {best_f1:.4f} (epoch stopped: {epoch})")

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        X_t = torch.from_numpy(X_test).float().to(device)
        mask_t = torch.from_numpy(mask_test).bool().to(device)
        for i in range(0, len(X_t), 256):
            logits = model(X_t[i:i + 256], pad_mask=mask_t[i:i + 256])
            preds.append(logits.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(preds)

    macro_f1_5 = f1_score(y_test, preds, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)
    present = [c for c in range(NUM_CLASSES) if c != 2]
    macro_f1_4 = f1_score(y_test, preds, average="macro", labels=present, zero_division=0)
    acc = (preds == y_test).mean()
    print(f"  CISS test: accuracy={acc:.4f}  macro-F1(5cls)={macro_f1_5:.4f}  "
          f"macro-F1(4cls excl. Head-on)={macro_f1_4:.4f}")
    return {"best_val_macro_f1": best_f1, "test_accuracy": acc,
            "test_macro_f1_5class": macro_f1_5, "test_macro_f1_4class": macro_f1_4}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_and_clean()

    results = {}

    # --- Ablation 1: no missingness indicators ---
    X, y, mask, ids, sources = build_sequences(df, SEQ_LEN, feature_cols=NO_INDICATOR_FEATURES)
    data = split_and_scale(X, y, mask, sources)
    results["no_missing_indicators"] = train_and_eval("no_missing_indicators (7 features, no *_missing)",
                                                        *data, device)

    # --- Ablation 2: steering forced to missing for every event ---
    df2 = df.copy()
    df2["steering_deg"] = 0.0
    df2["steering_deg_missing"] = 1.0
    X2, y2, mask2, ids2, sources2 = build_sequences(df2, SEQ_LEN)  # default FEATURE_COLS (11)
    data2 = split_and_scale(X2, y2, mask2, sources2)
    results["steering_zeroed"] = train_and_eval("steering_zeroed (steering_deg=0, steering_deg_missing=1 always)",
                                                  *data2, device)

    print(f"\n{'=' * 70}\nAblation summary (vs. full-feature Transformer baseline)\n{'=' * 70}")
    print("Baseline (full 11-feature Transformer, from results/comparison_summary.json):")
    print("  see evaluate.py output -- re-run evaluate.py after train.py for the current baseline number")
    for name, r in results.items():
        print(f"{name:<25} test_acc={r['test_accuracy']:.4f}  "
              f"test_macroF1_5cls={r['test_macro_f1_5class']:.4f}  "
              f"test_macroF1_4cls={r['test_macro_f1_4class']:.4f}")

    import json
    out_dir = results_dir("source")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ablations_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir / 'ablations_summary.json'}")


if __name__ == "__main__":
    main()

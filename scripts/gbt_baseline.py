"""
Gradient-boosted-tree baseline, on aggregated per-event features instead
of raw timestep sequences. With only a few thousand training events, a
deep sequence model has to learn temporal structure from very little
data; XGBoost on hand-summarized per-event statistics is a much lower-
variance model for this data regime and is a standard, expected baseline
comparison for small-dataset sequence classification.

For each event and each of the 11 UIR feature columns, computes 5
summary statistics over the REAL (non-padded) timesteps only: mean, std,
min, max, and the last real value (closest to impact, generally the most
informative single point for a crash). That's 11 x 5 = 55 engineered
features per event, built directly from the already-cleaned, already-
scaled tensors in models/tensors_mixed_split/ -- no re-reading the raw CSV.

Usage:
    .venv\\Scripts\\python.exe scripts\\gbt_baseline.py
"""

import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLASS_NAMES, FEATURE_COLS, NUM_CLASSES, RANDOM_STATE, RESULTS_DIR  # noqa: E402

TENSOR_DIR = Path(__file__).resolve().parent.parent / "models" / "tensors_mixed_split"

STATS = ["mean", "std", "min", "max", "last"]


def aggregate(X, mask):
    """X: (N, T, F), mask: (N, T) True=padding. Returns (N, F*len(STATS))."""
    N, T, F = X.shape
    valid = ~mask  # (N, T) True = real timestep
    feats = []
    for f in range(F):
        col = X[:, :, f]  # (N, T)
        # Set padded positions to NaN so nan-aware stats ignore them
        col_masked = np.where(valid, col, np.nan)
        with np.errstate(all="ignore"):
            mean = np.nanmean(col_masked, axis=1)
            std = np.nanstd(col_masked, axis=1)
            mn = np.nanmin(col_masked, axis=1)
            mx = np.nanmax(col_masked, axis=1)
        # last real value = value at the last True position in `valid`
        # (sequences are left-padded, so this is always index -1 when
        # there's at least one real timestep, per build_sequences.py)
        last = col[:, -1]
        stacked = np.stack([mean, std, mn, mx, last], axis=1)
        stacked = np.nan_to_num(stacked, nan=0.0)  # events with 0 real timesteps (shouldn't happen)
        feats.append(stacked)
    return np.concatenate(feats, axis=1)


def load(split):
    X = np.load(TENSOR_DIR / f"X_{split}.npy")
    y = np.load(TENSOR_DIR / f"y_{split}.npy")
    mask = np.load(TENSOR_DIR / f"mask_{split}.npy")
    return X, y, mask


def main():
    X_train, y_train, mask_train = load("train")
    X_val, y_val, mask_val = load("val")
    X_test, y_test, mask_test = load("test")

    # XGBoost doesn't need a separate val split for this comparison --
    # fold val into train for a slightly bigger training set, matching
    # what a from-scratch GBT pipeline would typically do.
    Xtr = np.concatenate([X_train, X_val])
    ytr = np.concatenate([y_train, y_val])
    masktr = np.concatenate([mask_train, mask_val])

    print("Aggregating per-event features...")
    Ftr = aggregate(Xtr, masktr)
    Fte = aggregate(X_test, mask_test)
    print(f"  train features: {Ftr.shape}, test features: {Fte.shape}")

    feature_names = [f"{col}_{stat}" for col in FEATURE_COLS for stat in STATS]

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
    )

    # small held-out slice of train for early stopping
    from sklearn.model_selection import train_test_split
    Ftr_fit, Ftr_es, ytr_fit, ytr_es = train_test_split(
        Ftr, ytr, test_size=0.1, random_state=RANDOM_STATE, stratify=ytr)

    model.fit(Ftr_fit, ytr_fit, eval_set=[(Ftr_es, ytr_es)], verbose=False)
    print(f"Best iteration: {model.best_iteration}")

    preds = model.predict(Fte)

    print(f"\n{'=' * 70}\nXGBoost baseline -- mixed-split test set (n={len(y_test)})\n{'=' * 70}")
    report = classification_report(y_test, preds, labels=list(range(NUM_CLASSES)),
                                    target_names=CLASS_NAMES, zero_division=0)
    print(report)
    acc = (preds == y_test).mean()
    macro_f1_5 = f1_score(y_test, preds, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)
    present = [c for c in range(NUM_CLASSES) if c != 2]
    macro_f1_4 = f1_score(y_test, preds, average="macro", labels=present, zero_division=0)
    print(f"Accuracy: {acc:.4f} | Macro-F1 (5cls): {macro_f1_5:.4f} | Macro-F1 (4cls excl. Head-on): {macro_f1_4:.4f}")

    # feature importance -- quick sanity check on what the tree model leans on
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print("\nTop 15 features by importance:")
    for i in top_idx:
        print(f"  {feature_names[i]:<30} {importances[i]:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "gbt_baseline_report.txt", "w") as f:
        f.write(f"test_accuracy={acc:.4f}\ntest_macro_f1_5class={macro_f1_5:.4f}\n"
                f"test_macro_f1_4class={macro_f1_4:.4f}\n\n{report}\n\nTop features:\n")
        for i in top_idx:
            f.write(f"  {feature_names[i]:<30} {importances[i]:.4f}\n")
    print(f"\nSaved to {RESULTS_DIR / 'gbt_baseline_report.txt'}")


if __name__ == "__main__":
    main()

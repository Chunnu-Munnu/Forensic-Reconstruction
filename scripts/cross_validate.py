"""
5-fold stratified cross-validation of the XGBoost baseline over ALL of
CISS, not just one static 70/30 slice. Motivation: the mixed-split test
set (Part 12) is only 456 events, and some classes are thin there
(Sideswipe n=27) -- a single split's accuracy carries real sampling
noise. This script instead partitions CISS into 5 stratified folds; for
each fold, ~80% of CISS + all of SynSHRP2 + all of BeamNG is the training
pool and the held-out 20% of CISS is the test fold. Every real CISS crash
is therefore used as a test example EXACTLY ONCE across the 5 folds,
giving a mean +- std accuracy AND a pooled classification report (all
1,520 CISS predictions concatenated) that is far more statistically
defensible than any single train/test split.

Also trains slightly more real-CISS-per-fold (80%) than the static mixed
split (70%), which should itself help a little.

Usage:
    .venv\\Scripts\\python.exe scripts\\cross_validate.py
"""

import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sequences import build_sequences, load_and_clean  # noqa: E402
from config import CLASS_NAMES, NUM_CLASSES, RANDOM_STATE, REPO_ROOT, SEQ_LEN, TEST_SOURCE  # noqa: E402
from gbt_baseline import STATS, aggregate  # noqa: E402

# This is the project's primary/headline result (documentation.txt Part
# 13) -- it gets its own top-level results/ subfolder rather than fitting
# under strict/mixed/random_split, since it isn't really any one of those
# static splits (it cross-validates over all of CISS).
RESULTS_DIR = REPO_ROOT / "results" / "cross_validation"


def main():
    df = load_and_clean()
    X, y, mask, ids, sources = build_sequences(df, SEQ_LEN)

    is_ciss = sources == TEST_SOURCE
    ciss_idx = np.where(is_ciss)[0]
    other_idx = np.where(~is_ciss)[0]
    print(f"\nCISS events: {len(ciss_idx)}  |  synthetic/naturalistic pool: {len(other_idx)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    all_true, all_pred = [], []
    fold_accs, fold_f1s = [], []

    for fold, (ciss_train_pos, ciss_test_pos) in enumerate(skf.split(ciss_idx, y[ciss_idx]), 1):
        ciss_train_idx = ciss_idx[ciss_train_pos]
        ciss_test_idx = ciss_idx[ciss_test_pos]

        train_pool_idx = np.concatenate([other_idx, ciss_train_idx])
        test_idx = ciss_test_idx

        N_tr, T_, F_ = X[train_pool_idx].shape
        scaler = StandardScaler().fit(X[train_pool_idx].reshape(-1, F_))

        def scale(idx):
            n = len(idx)
            return scaler.transform(X[idx].reshape(-1, F_)).reshape(n, T_, F_).astype(np.float32)

        X_train_s, X_test_s = scale(train_pool_idx), scale(test_idx)
        y_train, y_test = y[train_pool_idx], y[test_idx]
        mask_train, mask_test = mask[train_pool_idx], mask[test_idx]

        Ftr = aggregate(X_train_s, mask_train)
        Fte = aggregate(X_test_s, mask_test)

        # Inverse-frequency class weights, same rationale as the deep
        # models (README's bias-mitigation design) -- without this,
        # XGBoost's default objective optimizes overall accuracy and
        # all but ignores rare classes like Sideswipe (89/1520 CISS
        # events, and zero BeamNG examples at all).
        classes = np.arange(NUM_CLASSES)
        class_w = compute_class_weight("balanced", classes=classes, y=y_train)
        class_w_map = dict(zip(classes, class_w))
        sample_w_train = np.array([class_w_map[c] for c in y_train])

        Ftr_fit, Ftr_es, ytr_fit, ytr_es, w_fit, w_es = train_test_split(
            Ftr, y_train, sample_w_train, test_size=0.1,
            random_state=RANDOM_STATE, stratify=y_train)

        model = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=NUM_CLASSES,
            random_state=RANDOM_STATE, eval_metric="mlogloss",
            early_stopping_rounds=30,
        )
        model.fit(Ftr_fit, ytr_fit, sample_weight=w_fit,
                   eval_set=[(Ftr_es, ytr_es)], sample_weight_eval_set=[w_es], verbose=False)

        preds = model.predict(Fte)
        acc = (preds == y_test).mean()
        present = [c for c in range(NUM_CLASSES) if c != 2]
        f1 = f1_score(y_test, preds, average="macro", labels=present, zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        all_true.extend(y_test.tolist())
        all_pred.extend(preds.tolist())
        print(f"Fold {fold}/5: n_test={len(y_test)}  acc={acc:.4f}  macro-F1(4cls)={f1:.4f}  "
              f"best_iter={model.best_iteration}")

    fold_accs, fold_f1s = np.array(fold_accs), np.array(fold_f1s)
    print(f"\n{'=' * 70}\nCross-validated result across all 5 folds\n{'=' * 70}")
    print(f"Mean accuracy:        {fold_accs.mean():.4f} +/- {fold_accs.std():.4f}")
    print(f"Mean macro-F1(4cls):  {fold_f1s.mean():.4f} +/- {fold_f1s.std():.4f}")

    print(f"\n{'=' * 70}\nPOOLED report (all 1,520 CISS events, each predicted exactly once)\n{'=' * 70}")
    report = classification_report(all_true, all_pred, labels=list(range(NUM_CLASSES)),
                                    target_names=CLASS_NAMES, zero_division=0)
    print(report)
    pooled_acc = np.mean(np.array(all_true) == np.array(all_pred))
    pooled_f1_5 = f1_score(all_true, all_pred, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)
    present = [c for c in range(NUM_CLASSES) if c != 2]
    pooled_f1_4 = f1_score(all_true, all_pred, average="macro", labels=present, zero_division=0)
    print(f"Pooled accuracy: {pooled_acc:.4f} | Pooled macro-F1 (5cls): {pooled_f1_5:.4f} | "
          f"Pooled macro-F1 (4cls excl. Head-on): {pooled_f1_4:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "cross_validation_report.txt", "w") as f:
        f.write(f"5-fold stratified CV over all CISS events (n={len(ciss_idx)})\n")
        f.write(f"per-fold accuracy: {fold_accs.tolist()}\n")
        f.write(f"per-fold macro-F1(4cls): {fold_f1s.tolist()}\n")
        f.write(f"mean accuracy: {fold_accs.mean():.4f} +/- {fold_accs.std():.4f}\n")
        f.write(f"mean macro-F1(4cls): {fold_f1s.mean():.4f} +/- {fold_f1s.std():.4f}\n\n")
        f.write(f"pooled accuracy: {pooled_acc:.4f}\n")
        f.write(f"pooled macro-F1(5cls): {pooled_f1_5:.4f}\n")
        f.write(f"pooled macro-F1(4cls): {pooled_f1_4:.4f}\n\n")
        f.write(report)
    print(f"\nSaved to {RESULTS_DIR / 'cross_validation_report.txt'}")


if __name__ == "__main__":
    main()

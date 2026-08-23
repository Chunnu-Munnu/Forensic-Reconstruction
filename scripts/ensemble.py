"""
Heterogeneous soft-voting ensemble -- the project's best result
(documentation.txt Part 14).

WHAT IT IS
Three deliberately dissimilar members, soft-voted (their predicted class
probabilities are summed, then argmax'd):

  1. xgb_agg      XGBoost on the 55 aggregated features from gbt_baseline.py,
                  trained on the mixed pool (all non-CISS + the fold's CISS
                  training share) with non-CISS down-weighted.
  2. xgb_cissonly XGBoost on the 198 physics features from features.py,
                  trained on the fold's REAL CISS EVENTS ONLY. This member
                  exists for distribution diversity, not raw strength: on its
                  own it is the weakest member by accuracy (50.2%) but the
                  strongest by macro-F1 (0.433), because it never sees the
                  synthetic/staged domain at all.
  3. et_rich      ExtraTrees on the 198 physics features, mixed pool. A
                  different variance/bias profile from boosting -- bagged,
                  fully-grown, randomized-split trees.

Ensembling members that disagree for STRUCTURAL reasons (different feature
views, different training distributions, different tree-building algorithms)
is what makes the vote worth more than its parts; six near-identical XGBoost
variants were also tried and added almost nothing.

EVALUATION -- REPEATED cross-validation, and why
scripts/cross_validate.py uses ONE 5-fold CV pass (a single seed). That was
enough to rank models, but not to measure a 1-2 point change: per-fold
accuracy std is ~2-3 points, so a single 5-fold pass will happily show a
"gain" that is pure fold-assignment luck. This was not hypothetical -- during
development, richer features looked like +0.7pp on one seed and measured
-0.18pp once repeated over three. So this script runs the whole 5-fold CV
THREE times with different fold assignments (seeds 42, 7, 2024) = 15
train/test estimates, and reports mean +- std over all 15.

Every one of CISS's 1,520 real crashes is a held-out test event once per
seed, three times in total.

Usage:
    .venv\\Scripts\\python.exe scripts\\ensemble.py
    .venv\\Scripts\\python.exe scripts\\ensemble.py --seeds 42        (faster, 1 pass)

Rebuilds its own tensors from the raw CSV (~1-2 min), so it does not depend
on build_sequences.py or train.py having been run first.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sequences import build_sequences, load_and_clean  # noqa: E402
from config import (  # noqa: E402
    CLASS_NAMES, FEATURE_COLS, NUM_CLASSES, REPO_ROOT, SEQ_LEN, TEST_SOURCE,
)
from features import build_rich  # noqa: E402
from gbt_baseline import aggregate  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "ensemble"

# Class 2 (Head-on) has zero CISS events -- it is in the label space because
# the TRAINING pool has 1,204 of them (CTDB staged frontal tests, SynSHRP2),
# but it can never be a correct answer on the CISS test set. Macro-F1 is
# therefore always reported over the 4 realizable classes, exactly as every
# other script in this project does (see documentation.txt Part 2.4).
PRESENT = [c for c in range(NUM_CLASSES) if c != 2]

# Weight applied to every non-CISS training event, on top of the usual
# inverse-frequency class weight. The out-of-domain pool is large (10,593
# events vs ~1,216 real CISS per fold) and, left at full weight, it dominates
# the fit with a distribution the model is not actually tested on. Sweeping
# this (documentation.txt Part 14.2) showed the pool still HELPS -- dropping
# it entirely costs ~3 points of accuracy -- but it should inform the model,
# not outvote the real data.
DOMAIN_WEIGHT = 0.05


def fold_sample_weight(y_train, src_train, domain_weight=DOMAIN_WEIGHT):
    """Inverse-frequency class weight x domain weight (README's bias-mitigation
    rule: Final Sample Weight = Source Weight x Class Weight)."""
    present = np.unique(y_train)
    cw = dict(zip(present, compute_class_weight("balanced", classes=present, y=y_train)))
    class_w = np.array([cw.get(c, 1.0) for c in y_train], dtype=float)
    return class_w * np.where(src_train == TEST_SOURCE, 1.0, domain_weight)


def _expand(proba, classes):
    """Map a model's (n, k) proba over `classes` back onto the full 5 columns."""
    out = np.zeros((proba.shape[0], NUM_CLASSES))
    for j, c in enumerate(classes):
        out[:, int(c)] = proba[:, j]
    return out


def _fit_xgb(F_train, y_train, w_train, F_test, seed):
    """XGBoost needs contiguous 0..K-1 labels; a CISS-only pool has no class 2,
    so remap before fitting and expand back afterwards."""
    present = np.unique(y_train)
    remap = {c: i for i, c in enumerate(present)}
    y_m = np.array([remap[c] for c in y_train])

    F_fit, F_es, y_fit, y_es, w_fit, w_es = train_test_split(
        F_train, y_m, w_train, test_size=0.1, random_state=seed, stratify=y_m)
    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=len(present),
        random_state=seed, eval_metric="mlogloss",
        early_stopping_rounds=30, n_jobs=8)
    model.fit(F_fit, y_fit, sample_weight=w_fit, eval_set=[(F_es, y_es)],
              sample_weight_eval_set=[w_es], verbose=False)
    return _expand(model.predict_proba(F_test), present)


MEMBERS = ("xgb_agg", "xgb_cissonly", "et_rich")


def main(seeds):
    df = load_and_clean()
    X, y, mask, ids, sources = build_sequences(df, SEQ_LEN)

    print("\nBuilding physics + distributional features (features.py) ...")
    rich, rich_names = build_rich(X, mask, FEATURE_COLS)
    print(f"  {rich.shape[1]} features per event")

    ciss_idx = np.where(sources == TEST_SOURCE)[0]
    other_idx = np.where(sources != TEST_SOURCE)[0]
    print(f"CISS events: {len(ciss_idx)}  |  out-of-domain pool: {len(other_idx)}")

    n_feat = X.shape[2]
    scores = {m: {"acc": [], "f1": []} for m in MEMBERS}
    scores["ENSEMBLE"] = {"acc": [], "f1": []}
    pooled_true, pooled_pred = [], []  # first seed only, for the pooled report
    t0 = time.time()

    for si, seed in enumerate(seeds):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr_pos, te_pos) in enumerate(skf.split(ciss_idx, y[ciss_idx]), 1):
            ciss_train = ciss_idx[tr_pos]
            train_idx = np.concatenate([other_idx, ciss_train])
            test_idx = ciss_idx[te_pos]
            y_train, y_test = y[train_idx], y[test_idx]
            w_train = fold_sample_weight(y_train, sources[train_idx])

            # --- feature view 1: aggregated stats over SCALED sequences ---
            # (scaler fit on this fold's training pool only -- no test leakage)
            scaler = StandardScaler().fit(X[train_idx].reshape(-1, n_feat))

            def scaled(idx):
                return scaler.transform(
                    X[idx].reshape(-1, n_feat)).reshape(len(idx), X.shape[1], n_feat)

            agg_train = aggregate(scaled(train_idx), mask[train_idx])
            agg_test = aggregate(scaled(test_idx), mask[test_idx])

            # --- feature view 2: physics features on RAW sequences ---
            rich_train, rich_test = rich[train_idx], rich[test_idx]

            proba = {}
            proba["xgb_agg"] = _fit_xgb(agg_train, y_train, w_train, agg_test, seed)

            # CISS-only member: real crashes only, so no domain weighting
            w_ciss = fold_sample_weight(y[ciss_train], sources[ciss_train], 1.0)
            proba["xgb_cissonly"] = _fit_xgb(
                rich[ciss_train], y[ciss_train], w_ciss, rich_test, seed)

            et = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2,
                                      max_features="sqrt", random_state=seed, n_jobs=8)
            et.fit(rich_train, y_train, sample_weight=w_train)
            proba["et_rich"] = _expand(et.predict_proba(rich_test), et.classes_)

            for m in MEMBERS:
                pred_m = proba[m].argmax(1)
                scores[m]["acc"].append((pred_m == y_test).mean())
                scores[m]["f1"].append(f1_score(y_test, pred_m, average="macro",
                                                labels=PRESENT, zero_division=0))

            pred = sum(proba[m] for m in MEMBERS).argmax(1)
            acc = (pred == y_test).mean()
            f1 = f1_score(y_test, pred, average="macro", labels=PRESENT, zero_division=0)
            scores["ENSEMBLE"]["acc"].append(acc)
            scores["ENSEMBLE"]["f1"].append(f1)
            if si == 0:
                pooled_true.extend(y_test.tolist())
                pooled_pred.extend(pred.tolist())
            print(f"  seed {seed} fold {fold}/5: n_test={len(y_test)} "
                  f"acc={acc:.4f} macro-F1(4cls)={f1:.4f}  [{time.time()-t0:.0f}s]",
                  flush=True)

    n_est = len(seeds) * 5
    print(f"\n{'=' * 78}")
    print(f"REPEATED CROSS-VALIDATION -- seeds={list(seeds)}, 5 folds each (n={n_est})")
    print(f"{'=' * 78}")
    lines = []
    for name in list(MEMBERS) + ["ENSEMBLE"]:
        a = np.array(scores[name]["acc"])
        f = np.array(scores[name]["f1"])
        line = (f"{name:<16} accuracy={a.mean():.4f} +/- {a.std():.4f}   "
                f"macro-F1(4cls)={f.mean():.4f} +/- {f.std():.4f}")
        print(line)
        lines.append(line)

    print(f"\n{'=' * 78}")
    print(f"POOLED report -- seed {seeds[0]}, all {len(pooled_true)} CISS events, "
          f"each predicted exactly once")
    print(f"{'=' * 78}")
    report = classification_report(pooled_true, pooled_pred,
                                   labels=list(range(NUM_CLASSES)),
                                   target_names=CLASS_NAMES, zero_division=0)
    print(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "ensemble_report.txt"
    with open(out, "w") as fh:
        fh.write(f"Soft-voting ensemble: {' + '.join(MEMBERS)}\n")
        fh.write(f"Repeated stratified CV: seeds={list(seeds)}, 5 folds each "
                 f"(n={n_est} train/test estimates)\n")
        fh.write(f"Non-CISS domain weight: {DOMAIN_WEIGHT}\n")
        fh.write(f"CISS events: {len(ciss_idx)} | out-of-domain pool: {len(other_idx)}\n")
        fh.write(f"Features: {rich.shape[1]} physics+distributional, "
                 f"{agg_train.shape[1]} aggregated\n\n")
        fh.write("\n".join(lines) + "\n\n")
        fh.write(f"Per-fold ensemble accuracy: "
                 f"{[round(float(v), 4) for v in scores['ENSEMBLE']['acc']]}\n")
        fh.write(f"Per-fold ensemble macro-F1(4cls): "
                 f"{[round(float(v), 4) for v in scores['ENSEMBLE']['f1']]}\n\n")
        fh.write(f"POOLED report (seed {seeds[0]}, every CISS event tested once)\n")
        fh.write(report)
    print(f"Saved to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024],
                    help="fold-assignment seeds; each runs a full 5-fold CV pass")
    args = ap.parse_args()
    main(tuple(args.seeds))

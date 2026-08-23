"""
Phase 2, Step 1: turn the flat per-timestep UIR table into fixed-length
(N, T, F) sequence tensors that a PyTorch model can consume, and split
them by data source into train/val/test.

Run once per split mode (takes ~1-2 minutes on the full 865k-row CSV):
    .venv\\Scripts\\python.exe scripts\\build_sequences.py --split-mode source
    .venv\\Scripts\\python.exe scripts\\build_sequences.py --split-mode mixed
    .venv\\Scripts\\python.exe scripts\\build_sequences.py --split-mode random

Produces models/<split>/tensors/*.npy, models/<split>/feature_scaler.pkl,
models/<split>/feature_cols.json, models/<split>/test_event_ids.json, and
a printed data-quality report, where <split> is "strict", "mixed", or
"random_split" depending on --split-mode (see config.SPLIT_DIR_NAMES).
Every later script (train.py, evaluate.py, ablations.py) reads from
models/<split>/tensors/ instead of the raw CSV, so this is the only
script that needs pandas.

Two Phase-1 data bugs are corrected here rather than upstream (the raw
processed CSVs are left untouched -- see documentation.txt Part 3.3/8 for
the full investigation):

1. CISS rows store their pre-crash time axis in `PTIME`, not the unified
   `t` column (`t` is 100% NaN for every CISS row, `PTIME` is 100% NaN for
   every SynSHRP2/BeamNG row). Naively sorting/grouping by `t` alone
   silently drops all 1,520 CISS events -- which happen to be the entire
   test set. Fixed by coalescing: t_unified = t if present else PTIME.
2. 1,538 SynSHRP2 events (27% of SynSHRP2) contain duplicate rows for the
   same (event_id, t) pair -- a many-to-one join fan-out somewhere in the
   Phase 1 SynSHRP2 merge. Left unfixed, the worst event reports 443 rows
   for what is actually 100 unique timesteps, which would have silently
   set an inflated, wrong sequence length budget for the whole dataset.
   Fixed by averaging duplicate (event_id, t_unified) rows before
   building sequences.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    CISS_TRAIN_FRACTION, CLASS_NAMES, DATA_CSV, FEATURE_COLS, INDICATOR_FOR,
    NUM_CLASSES, PHYSICAL_BOUNDS, RANDOM_STATE, RAW_FEATURE_COLS,
    SEQ_LEN, SOURCE_WEIGHTS, TEST_SOURCE, TRAIN_SOURCES, model_dir, tensor_dir,
)


def load_and_clean():
    print(f"Loading {DATA_CSV} ...")
    df = pd.read_csv(DATA_CSV, low_memory=False)
    print(f"  {len(df):,} rows, {df['event_id'].nunique():,} unique events")

    # --- Bug fix 1: unify the time axis across sources ---
    df["t_unified"] = df["t"].fillna(df["PTIME"])
    assert df["t_unified"].isna().sum() == 0, "found rows with neither t nor PTIME set"

    # --- Bug fix 2 (INFERRED, not verified against raw source -- see
    # documentation.txt Part 9.3): SynSHRP2's speed_kmh looks like it was
    # converted twice. SynSHRP2_Cleaning_Explanation.md documents
    # `speed_kmh = Speed * 3.6` assuming the source JSON's `Speed` field
    # was in m/s. But the resulting SHRP2 speed_kmh distribution has a
    # median of 167.7 km/h and a 90th percentile of 355.3 km/h -- not
    # plausible for a naturalistic-driving study. Dividing the whole
    # column by 3.6 gives a median of 46.6 km/h and a 90th percentile of
    # 98.7 km/h, which lines up closely with CISS's real-crash speed
    # distribution (median 62.0, 90th 119.0 km/h) and with BeamNG's
    # (median 62.1, 90th 90.1 km/h). This strongly suggests `Speed` in
    # the original SynSHRP2 JSON was already in km/h and got multiplied
    # by 3.6 a second time during Phase 1. The raw per-event Kinematic
    # JSON files are not present in this repo (only the Tabular_records.csv
    # metadata is), so this cannot be verified against source -- it is
    # applied here as a well-evidenced statistical correction, flagged
    # explicitly so it can be checked against the original SynSHRP2 files
    # if/when they become available. ---
    shrp2_mask = df["source"] == "SHRP2"
    print(f"  correcting suspected double speed conversion for "
          f"{shrp2_mask.sum():,} SynSHRP2 rows (speed_kmh /= 3.6)")
    df.loc[shrp2_mask, "speed_kmh"] = df.loc[shrp2_mask, "speed_kmh"] / 3.6

    # --- Bug fix 3: sanitize un-decoded sentinel / corrupted sensor
    # values. See config.PHYSICAL_BOUNDS for the full rationale and the
    # specific corrupted values that forced this (accel_long up to
    # 425,115 m/s^2, steering_deg up to 81,779 degrees, etc.). Anything
    # outside the documented physical bound becomes NaN -- handled by the
    # exact same missing-indicator + zero-fill mechanism as a genuinely
    # unrecorded sensor, which is the correct semantics: a corrupted
    # reading and an absent reading are both "we don't actually know this
    # value," and the model should treat them identically. ---
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        bad = ~df[col].between(lo, hi) & df[col].notna()
        n_bad = int(bad.sum())
        if n_bad:
            print(f"  sanitized {n_bad:,} out-of-range {col} values "
                  f"(outside [{lo}, {hi}]) -> NaN")
        df.loc[bad, col] = np.nan

    # brake_active must be exactly 0.0 or 1.0 (documented as Binary in
    # CISS_DATA_CODES_GUIDE.md); anything else (e.g. CISS rows carrying
    # values like 1200.0, 6100.0, up to 131,070.0 -- clearly an
    # un-decoded raw code, not a brake state) is invalid.
    bad_brake = ~df["brake_active"].isin([0.0, 1.0]) & df["brake_active"].notna()
    n_bad_brake = int(bad_brake.sum())
    if n_bad_brake:
        print(f"  sanitized {n_bad_brake:,} non-binary brake_active values -> NaN")
    df.loc[bad_brake, "brake_active"] = np.nan

    # --- Missingness indicators (brake_active/yaw_rate never got one in
    # Phase 1; throttle_pct/steering_deg already have throttle_missing /
    # steering_missing but we recompute them here for consistency instead
    # of trusting the pre-existing columns, since they're cheap to
    # recompute and this removes any doubt about how they were derived). ---
    for col in INDICATOR_FOR:
        df[f"{col}_missing"] = df[col].isna().astype(np.float32)

    # --- Bug fix 2: dedupe rows sharing (event_id, t_unified). Average
    # the numeric feature columns; keep the first value for identity
    # columns (they're identical across the duplicate rows by construction).
    value_cols = RAW_FEATURE_COLS + [f"{c}_missing" for c in INDICATOR_FOR]
    id_cols = ["source", "source_year", "crash_class", "is_crash"]
    agg = {c: "mean" for c in value_cols}
    agg.update({c: "first" for c in id_cols})

    before = len(df)
    df = df.groupby(["event_id", "t_unified"], as_index=False).agg(agg)
    print(f"  deduped {before - len(df):,} duplicate (event_id, t) rows "
          f"({before:,} -> {len(df):,})")

    # Fill NaN in raw feature columns with 0 -- safe ONLY because the
    # matching *_missing indicator already recorded that this was absent.
    # Never do this for a column without a paired indicator.
    for col in RAW_FEATURE_COLS:
        df[col] = df[col].fillna(0.0)

    return df


def build_sequences(df, seq_len, feature_cols=None):
    feature_cols = feature_cols if feature_cols is not None else FEATURE_COLS
    events = df.sort_values("t_unified").groupby("event_id")
    F = len(feature_cols)

    X_list, y_list, mask_list, ids, sources = [], [], [], [], []
    dropped_unlabeled = 0

    for eid, grp in events:
        label = grp["crash_class"].iloc[0]
        if pd.isna(label):
            dropped_unlabeled += 1
            continue

        feats = grp[feature_cols].to_numpy(dtype=np.float32)
        n = len(feats)

        seq = np.zeros((seq_len, F), dtype=np.float32)
        mask = np.ones(seq_len, dtype=bool)  # True = padding position

        if n >= seq_len:
            seq[:] = feats[-seq_len:]         # keep timesteps closest to impact
            mask[:] = False
        else:
            seq[seq_len - n:] = feats          # left-pad
            mask[seq_len - n:] = False

        X_list.append(seq)
        y_list.append(int(label))
        mask_list.append(mask)
        ids.append(eid)
        sources.append(grp["source"].iloc[0])

    print(f"  built {len(X_list):,} labeled sequences "
          f"(dropped {dropped_unlabeled} events with unmapped/missing crash_class)")

    return (np.stack(X_list), np.array(y_list, dtype=np.int64),
            np.stack(mask_list), np.array(ids), np.array(sources))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN,
                         help="override config.SEQ_LEN, for experimentation "
                              "(e.g. --seq-len 50 to match CISS's short pre-impact window)")
    parser.add_argument("--split-mode", choices=["source", "random", "mixed"], default="source",
                         help="'source' (default) = strict zero-shot split, CISS held out "
                              "entirely. 'random' = diagnostic split, stratified random "
                              "70/15/15 across all sources pooled together. 'mixed' = "
                              "PRIMARY split: CISS_TRAIN_FRACTION of CISS enters the "
                              "train/val pool alongside SynSHRP2+BeamNG, the rest stays "
                              "held out as a genuinely real-world test set (documentation.txt "
                              "Part 12).")
    args = parser.parse_args()
    seq_len = args.seq_len

    TENSOR_DIR = tensor_dir(args.split_mode)
    OUT_MODEL_DIR = model_dir(args.split_mode)
    TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean()
    X, y, pad_mask, event_ids, sources = build_sequences(df, seq_len)
    print(f"\nFinal tensor: X={X.shape}, y={y.shape}, pad_mask={pad_mask.shape}")
    print("Class distribution (all events):",
          {CLASS_NAMES[c]: int((y == c).sum()) for c in range(NUM_CLASSES)})

    if args.split_mode == "source":
        # --- Hard source-based split (see documentation.txt Part 4.3): the
        # PRIMARY, README-mandated evaluation protocol. CISS (real crashes)
        # held out entirely; SynSHRP2 + BeamNG for train/val. ---
        is_train_pool = np.isin(sources, TRAIN_SOURCES)
        is_test = sources == TEST_SOURCE
        assert is_train_pool.sum() + is_test.sum() == len(sources), \
            "found a source that is neither a train source nor the test source"

        train_pool_idx = np.where(is_train_pool)[0]
        test_idx = np.where(is_test)[0]

        train_idx, val_idx = train_test_split(
            train_pool_idx, test_size=0.15, random_state=RANDOM_STATE,
            stratify=y[train_pool_idx],
        )
    elif args.split_mode == "mixed":
        # --- PRIMARY split (documentation.txt Part 12): CISS_TRAIN_FRACTION
        # of CISS enters the train/val pool alongside SynSHRP2+BeamNG,
        # stratified by crash_class; the remaining CISS events are held
        # out as a genuinely real-world test set. This directly addresses
        # the sim-to-real generalization gap found under the strict
        # 'source' split (documentation.txt Part 9.6) by giving the model
        # real examples of what CISS's short EDR windows and sensor
        # distributions actually look like, at the cost of the stronger
        # "zero real crashes seen during training" claim. Head-on (class
        # 2) still has ZERO CISS events at any split fraction -- that
        # limitation is unchanged (see Part 2.4). ---
        is_ciss = sources == TEST_SOURCE
        ciss_idx = np.where(is_ciss)[0]
        other_idx = np.where(~is_ciss)[0]

        ciss_train_idx, ciss_test_idx = train_test_split(
            ciss_idx, train_size=CISS_TRAIN_FRACTION, random_state=RANDOM_STATE,
            stratify=y[ciss_idx],
        )

        train_pool_idx = np.concatenate([other_idx, ciss_train_idx])
        test_idx = ciss_test_idx

        train_idx, val_idx = train_test_split(
            train_pool_idx, test_size=0.15, random_state=RANDOM_STATE,
            stratify=y[train_pool_idx],
        )
        print(f"  CISS split: {len(ciss_train_idx):,} into train/val pool, "
              f"{len(ciss_test_idx):,} held out for test "
              f"(fraction={CISS_TRAIN_FRACTION})")
    else:
        # --- Stratified RANDOM split across all sources pooled together
        # (SECONDARY, diagnostic evaluation -- see documentation.txt Part
        # 9.4). Answers a different question than the source-based split:
        # "do the features and architecture support this classification
        # task at all when train and test come from the same
        # distribution?" This is what a conventional ML paper would
        # report by default; it is NOT the project's real-world
        # generalization claim, which is exactly why it is kept separate
        # and off by default (--split-mode random to enable). ---
        all_idx = np.arange(len(sources))
        train_idx, temp_idx = train_test_split(
            all_idx, test_size=0.30, random_state=RANDOM_STATE, stratify=y[all_idx])
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, random_state=RANDOM_STATE, stratify=y[temp_idx])

    test_label = {
        "source": "(CISS, held out from training entirely)",
        "mixed": f"(CISS, {1 - CISS_TRAIN_FRACTION:.0%} held out; rest used for training)",
        "random": "(random pooled slice, same distribution as train)",
    }[args.split_mode]
    print(f"\nSplit mode: {args.split_mode} | train={len(train_idx):,}  "
          f"val={len(val_idx):,}  test={len(test_idx):,} {test_label}")

    test_classes_present = sorted(set(y[test_idx].tolist()))
    missing_from_test = [c for c in range(NUM_CLASSES) if c not in test_classes_present]
    if missing_from_test:
        names = [CLASS_NAMES[c] for c in missing_from_test]
        print(f"  WARNING: class(es) {names} have ZERO examples in the test "
              f"split -- their test-set metrics are not evaluable "
              f"(see documentation.txt Part 2.4).")

    # --- Normalize: fit on train only ---
    N_tr, T_, F_ = X[train_idx].shape
    scaler = StandardScaler().fit(X[train_idx].reshape(-1, F_))

    def scale(idx):
        n = len(idx)
        return scaler.transform(X[idx].reshape(-1, F_)).reshape(n, T_, F_).astype(np.float32)

    X_train, X_val, X_test = scale(train_idx), scale(val_idx), scale(test_idx)
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    mask_train, mask_val, mask_test = pad_mask[train_idx], pad_mask[val_idx], pad_mask[test_idx]
    src_train = sources[train_idx]

    # --- Per-sample weights for the train set: class_weight x source_weight ---
    class_w = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train)
    class_w_map = {c: w for c, w in enumerate(class_w)}
    sample_w = np.array([class_w_map[label] * SOURCE_WEIGHTS.get(src, 1.0)
                          for label, src in zip(y_train, src_train)], dtype=np.float32)

    print("\nPer-class weight (inverse-frequency, train split only):")
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]:<15} weight={class_w_map[c]:.3f}  "
              f"n_train={int((y_train == c).sum())}")

    # --- Save everything ---
    np.save(TENSOR_DIR / "X_train.npy", X_train)
    np.save(TENSOR_DIR / "y_train.npy", y_train)
    np.save(TENSOR_DIR / "mask_train.npy", mask_train)
    np.save(TENSOR_DIR / "sample_weight_train.npy", sample_w)

    np.save(TENSOR_DIR / "X_val.npy", X_val)
    np.save(TENSOR_DIR / "y_val.npy", y_val)
    np.save(TENSOR_DIR / "mask_val.npy", mask_val)

    np.save(TENSOR_DIR / "X_test.npy", X_test)
    np.save(TENSOR_DIR / "y_test.npy", y_test)
    np.save(TENSOR_DIR / "mask_test.npy", mask_test)

    # Metadata is written into the same models/<split>/ directory as the
    # tensors and (later) the checkpoints -- no more filename tagging
    # needed now that each split has its own directory.
    dump(scaler, OUT_MODEL_DIR / "feature_scaler.pkl")
    with open(OUT_MODEL_DIR / "feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    with open(OUT_MODEL_DIR / "test_event_ids.json", "w") as f:
        json.dump(event_ids[test_idx].tolist(), f)

    # Fixed SHAP background sample for Phase 3 (200 random train sequences)
    rng = np.random.default_rng(RANDOM_STATE)
    bg_idx = rng.choice(len(X_train), size=min(200, len(X_train)), replace=False)
    np.save(OUT_MODEL_DIR / "shap_background_X.npy", X_train[bg_idx])
    np.save(OUT_MODEL_DIR / "shap_background_mask.npy", mask_train[bg_idx])

    print(f"\nSaved tensors to {TENSOR_DIR}")
    print(f"Saved scaler/metadata to {OUT_MODEL_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()

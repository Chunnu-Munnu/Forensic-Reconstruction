# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Phase 1 (data engineering, producing `Dataset/data/training_dataset_final.csv`) and Phase 2 (sequence model training) are both complete. Phase 2's code lives in `scripts/` (`config.py`, `build_sequences.py`, `models.py`, `train.py`, `evaluate.py`, `ablations.py`, `random_split_diagnostic.py`) — read `docs/documentation.txt` before touching any of it; it explains every design decision, three real Phase 1 data-quality bugs found and fixed during Phase 2, and the exact results. Phases 3-5 (explainability, physics validation, 3D reconstruction) are out of scope and not implemented — see `docs/documentation.txt` Part 5.

Python environment: a project-local venv at `.venv/` (git-ignored), Python 3.11, built per `docs/documentation.txt` Part 8.1 — the system's default Python (3.14) has a broken MINGW-built numpy that segfaults; do not use it. Rebuild with:
```
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install numpy pandas scikit-learn matplotlib seaborn joblib tqdm
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```
`Dataset/requirements.txt` (UTF-16 encoded — read with `encoding="utf-16"` / `-Encoding Unicode`) is Phase 1's original environment spec and is not what `.venv` was built from.

Run order: `scripts/build_sequences.py` (builds tensors from the CSV, ~1-2 min) → `scripts/train.py` (trains all 3 models, ~1-2 min each on an RTX 3050, likely fine on CPU too given how small this dataset/model are) → `scripts/evaluate.py` (evaluates on the held-out test set). `scripts/ablations.py` and `scripts/random_split_diagnostic.py` are independent secondary experiments, run after the primary pipeline.

All three of the above scripts take `--split-mode {source,mixed,random}` (build_sequences.py) or `{source,mixed}` (train.py, evaluate.py) — `mixed` fine-tunes on 70% of real CISS crashes, tests on the remaining 456 (`docs/documentation.txt` Part 12); `source` is the stricter zero-real-crashes-in-training comparison (Part 9); `random` (build_sequences.py only) is the in-distribution sanity check consumed by `scripts/random_split_diagnostic.py`. Each split mode gets its own directory — `models/strict/`, `models/mixed/`, `models/random_split/`, mirrored under `results/` — so the three never overwrite each other; checkpoints/scaler/metadata/history live under `models/<split>/`, evaluate.py's outputs (confusion matrices, prediction arrays, summaries) under `results/<split>/`. Always pass matching `--split-mode` to `train.py` and `evaluate.py`. `scripts/config.py`'s `model_dir()`/`results_dir()`/`tensor_dir()` helpers are the single source of truth for these paths — use them rather than constructing paths by hand.

**The actual primary/recommended result is `scripts/cross_validate.py`** (`docs/documentation.txt` Part 13, needs `pip install xgboost`): a class-weighted XGBoost model on aggregated per-event features (not raw sequences — see `scripts/gbt_baseline.py`), evaluated with 5-fold CV over all 1,520 real CISS crashes so every one is tested exactly once. It beats every deep model and reports mean±std instead of one split's number. Run it after `build_sequences.py --split-mode mixed`; it rebuilds its own tensors from the raw CSV internally (~1-2 min) so it doesn't depend on train.py having been run first.

## Project goal (Phase 1)

Build a **Unified Intermediate Representation (UIR)**: a single per-timestep schema that multiple heterogeneous crash datasets (CISS, SynSHRP2, IGLAD PCM, BeamNG synthetic) are normalized into, for training a crash-classification/reconstruction model in later phases. Full rationale is in `README.md`.

UIR columns (see `Dataset/data/training_dataset_final.csv` header for the exact current schema):
```
event_id, source, source_year, crash_class, is_crash, t,
speed_kmh, accel_long, accel_lat, brake_active, throttle_pct,
steering_deg, yaw_rate, engine_rpm, abs_esc_active,
throttle_missing, steering_missing, ...
```
Plus source-specific raw fields carried through per dataset (e.g. CISS keeps `CASEID`, `VEHNO`, `CRASHTYPE`, `DVTOTAL`/`DVLONG`/`DVLAT`, `SURFCOND`, `LIGHTCOND`, `WEATHER`, `MODELYR`).

**Core design rule: never impute missing sensor values with arbitrary constants.** A field either has its real measured value, or is `NaN` with a paired `*_missing` indicator column (e.g. `throttle_missing`, `steering_missing`). This lets the model distinguish "sensor unavailable" from "sensor measured zero" — do not silently fill NaNs with 0 or a mean when adding new processing code.

5-class crash taxonomy used everywhere (`crash_class`): `0=Rear-end, 1=Angle/intersection, 2=Head-on, 3=Single-vehicle, 4=Sideswipe`. Unmapped/unknown crash types are dropped, not bucketed into an "other" class.

## Data layout

- `Dataset/data/raw/<SOURCE>/<year>/*.CSV` — raw NHTSA CISS relational tables (one file per year, 2019–2024; note 2019 files are `UPPERCASE.CSV`, 2020+ are `lowercase.csv`, and 2024 adds extra tables like `NONMOTORIST`, `NMKINEMATICS`, plus a `FORMAT24.sas` format spec). CISS is a relational schema — a single crash event is assembled by joining multiple tables (`CRASH`, `GV`, `EDRPRECRASH`, `EDRSUMM`, `OCC`, etc.) on shared keys like `CASEID`/`VEHNO`.
- `Dataset/data/raw/SynSHRP2/Tabular_records.csv` — SynSHRP2 event metadata (per-event kinematic JSON files referenced in the cleaning doc are not present in this repo yet).
- `Dataset/data/processed/CISS/<year>/CISS_<year>_clean.csv` — one cleaned/UIR-mapped file per CISS year.
- `Dataset/data/processed/CISS/CISS_DATA_CODES_GUIDE.md` — **read this before touching any CISS field**: explains the 999="unknown" vs 0="measured" distinction, CRASHTYPE/SURFCOND/LIGHTCOND/WEATHER/MANEUVER code tables, and the raw-field → UIR-column mapping.
- `Dataset/data/processed/SynSHRP2/SynSHRP2_clean.csv` and `SynSHRP2_Cleaning_Explanation.md` — the SynSHRP2 cleaning steps (timestamp normalization relative to impact, unit conversions g→m/s² and m/s→km/h, pre-crash window trim to [-20s, 0s], incident-type → crash_class mapping) with full rationale; read before modifying SynSHRP2 processing.
- `Dataset/data/training_dataset_final.csv` — the merged CISS + SynSHRP2 (+ BeamNG per README) output dataset in UIR schema. Per the README, real-world CISS events are reserved for final evaluation only; SynSHRP2 and BeamNG synthetic data are training-only.

## Working with this data

- Do not "fix" or reinterpret 999 codes in CISS fields without checking `CISS_DATA_CODES_GUIDE.md` first — 999 is a distinct semantic ("unknown"), not an outlier to clean.
- When adding a new data source, follow the same pattern as SynSHRP2: produce output strictly in the shared UIR column schema, add `*_missing` indicator columns for any field the source doesn't record, tag `source` and `source_year`, and prefix `event_id` with a source tag to avoid ID collisions (e.g. `SHRP2_<id>`, `CISS_<year>_<caseid>_V<vehno>`).
- CSV files are large (CISS raw tables, `training_dataset_final.csv`); prefer streaming/chunked reads (pandas `chunksize`) or targeted column selection over loading everything into memory at once when writing new processing scripts.
- Source and class imbalance are handled via multiplicative sample weighting (`source_weight × class_weight`), per the README's Bias Mitigation Strategy — keep this in mind if implementing training-time weighting logic.

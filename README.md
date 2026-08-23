# Phase 1: Unified Multi-Source Crash Dataset Construction

## Overview

The objective of Phase 1 is to construct a unified, high-quality crash
dataset by integrating multiple heterogeneous crash data sources into a
common representation. Existing crash datasets differ significantly in
structure, sensor availability, sampling frequency, variable naming
conventions, and class distributions. These inconsistencies make it
difficult to directly train a robust machine learning model across
multiple datasets.

To overcome this challenge, Phase 1 focuses on developing a standardized
preprocessing pipeline that converts all datasets into a common
**Unified Intermediate Representation (UIR)** while preserving their
temporal characteristics and minimizing dataset-specific bias.

The resulting dataset serves as the foundation for all subsequent model
development and evaluation.

------------------------------------------------------------------------

# Motivation

Current publicly available crash datasets each have unique strengths but
also significant limitations.

-   **CISS** provides high-quality real-world crash investigations with
    Event Data Recorder (EDR) information but has limited sample
    diversity.
-   **SynSHRP2** provides large-scale synthetic naturalistic driving
    data but lacks certain vehicle control signals available in other
    datasets.
-   **IGLAD PCM** offers reconstructed international crash cases with
    detailed vehicle dynamics but follows a completely different data
    format.
-   Rare crash scenarios remain underrepresented across all publicly
    available datasets.

Training a model on any single dataset introduces domain bias and limits
generalization.

Instead of relying on one source, this project combines multiple
complementary datasets into a unified learning framework while
explicitly accounting for differences between data sources.

------------------------------------------------------------------------

# Design Philosophy

Rather than forcing every dataset into an identical structure through
aggressive preprocessing, the pipeline preserves available information
while explicitly recording missing variables.

Missing sensor values are never replaced with arbitrary constants during
dataset construction. Instead:

-   Available measurements are standardized.
-   Missing measurements remain as `NaN`.
-   Dedicated missing-value indicators are generated for every optional
    feature.

This approach allows downstream learning algorithms to distinguish
between "sensor unavailable" and "sensor measured zero," reducing
information loss.

------------------------------------------------------------------------

# Data Sources

## 1. CISS (Crash Investigation Sampling System)

-   Source: National Highway Traffic Safety Administration (NHTSA)
-   Years: 2019--2023
-   Role:
    -   Real-world crash benchmark
    -   Event Data Recorder (EDR) measurements
    -   Final unseen evaluation dataset

## 2. SynSHRP2

Synthetic benchmark derived from the Second Strategic Highway Research
Program (SHRP2).

Role:

-   Large-scale driving trajectories
-   Naturalistic vehicle behavior
-   Additional temporal diversity

## 3. IGLAD PCM

International in-depth reconstructed crash database.

Role:

-   Detailed vehicle dynamics
-   International crash scenarios
-   High-quality reconstructed trajectories

## 4. BeamNG Synthetic Data

Custom crash scenarios generated using BeamNGpy.

Role:

-   Rare crash augmentation
-   Long-tail crash classes
-   Controlled scenario generation

BeamNG-generated data is used exclusively during training and is never
included in the final evaluation set.

------------------------------------------------------------------------

# Phase 1 Objectives

1.  Organize all raw datasets into a consistent directory structure.
2.  Explore dataset schemas before implementation.
3.  Merge CISS relational tables into complete crash events.
4.  Parse SynSHRP2 temporal driving records.
5.  Parse IGLAD XML dynamics files.
6.  Convert all measurements into consistent physical units.
7.  Build a Unified Intermediate Representation (UIR).
8.  Remove incomplete or low-quality events.
9.  Correct inter-dataset imbalance using source-aware weighting.
10. Correct class imbalance using inverse-frequency weighting.
11. Reserve real-world CISS events exclusively for final testing.
12. Augment underrepresented crash classes using BeamNG simulation.
13. Normalize and resample all sequences to a uniform temporal
    resolution.

------------------------------------------------------------------------

# Unified Intermediate Representation (UIR)

Each row corresponds to a single timestep within a crash event.

Core variables:

-   Event ID
-   Data source
-   Crash type
-   Timestamp
-   Vehicle speed
-   Longitudinal acceleration
-   Lateral acceleration
-   Brake status
-   Throttle position
-   Steering angle
-   Yaw rate
-   Delta-V
-   Missing-value indicators

------------------------------------------------------------------------

# Bias Mitigation Strategy

Two complementary weighting strategies are employed.

## Source Weighting

Each dataset receives an inverse proportional weight based on its
contribution to the total number of events.

## Class Weighting

Inverse-frequency class weighting is applied to underrepresented crash
categories.

## Final Sample Weight

**Final Sample Weight = Source Weight × Class Weight**

These weights are incorporated into the model's weighted cross-entropy
loss.

------------------------------------------------------------------------

# Data Quality Control

-   Remove events shorter than 20 timesteps.
-   Remove events with more than 30% missing values in core variables.
-   Standardize units across datasets.
-   Verify temporal consistency.
-   Preserve missing sensor information with explicit indicators.

------------------------------------------------------------------------

# Expected Output

    phase1_uir_clean.parquet

Expected characteristics:

-   Approximately 14,000--15,000 crash sequences.
-   Multi-source integration.
-   Source-tagged events.
-   Crash-type labels.
-   Sample weights.
-   Uniform temporal resolution (10 Hz).
-   Ready for Phase 2 model training.

------------------------------------------------------------------------

# Deliverable

Phase 1 establishes the complete data engineering pipeline required for
robust crash prediction. All datasets are transformed into a common
representation while preserving temporal information, handling missing
sensors explicitly, correcting dataset imbalance, and maintaining an
unbiased evaluation protocol.

This standardized dataset forms the foundation for the multi-task crash
intelligence framework developed in subsequent phases.

------------------------------------------------------------------------

# Phase 2: Model Training and Evaluation (Complete)

Phase 2 builds and evaluates crash-type classifiers on top of Phase 1's
unified dataset. Full technical detail -- every design decision, every bug
found and fixed, every result, and exact reproduction commands -- is in
`docs/documentation.txt`. This section is a summary.

## Data: what Phase 1 delivered, and what changed since

Phase 1 produced `Dataset/data/training_dataset_final.csv`: heterogeneous
crash datasets normalized into one per-timestep Unified Intermediate
Representation (UIR), with `*_missing` indicator columns instead of
silently imputed values. It originally merged three sources.

Phase 2 added a fourth and re-ran everything:

| Source | Events | Kind | Role |
| --- | ---: | --- | --- |
| SynSHRP2 | 5,701 | Real naturalistic driving | Training |
| CTDB | 3,442 | Real staged NHTSA crash tests | Training |
| CISS | 1,520 | Real crash investigations (EDR) | **Evaluation target** |
| BeamNG | 1,450 | Synthetic simulation | Training |
| **Total** | **12,113** | | |

**CTDB** (NHTSA Crash Test Database) is new in Phase 2 and is the single
biggest data change: 3,442 real staged crash tests pulled from NHTSA's
live API (`scripts/fetch_crash_test_index.py`,
`scripts/download_crash_test_db.py`) and converted into UIR rows
(`scripts/integrate_crash_test_db.py`). It was added specifically because
Rear-end was the weakest class and BeamNG generates zero rear-end
scenarios. It worked: pooled Rear-end recall went from 0.16 to ~0.52.

Adding CTDB grew the dataset from 8,671 to 12,113 events and moved
cross-validated accuracy from 49.2% to 51.7%.

Three Phase-1 data defects were found and fixed while building this:
duplicate timestep rows, a likely double unit conversion in SynSHRP2's
speed field, and un-decoded NHTSA sentinel codes contaminating several
CISS EDR fields (`accel_long` readings up to 43,000 g). See
`docs/documentation.txt` Part 9.2.

## What was built

-   `scripts/build_sequences.py` -- flat CSV to fixed-length
    `(N, 160, 11)` tensors; three split modes
    (`--split-mode source|mixed|random`).
-   `scripts/models.py`, `train.py`, `evaluate.py` -- three PyTorch
    sequence architectures (BiLSTM, BiLSTM+attention, Transformer
    encoder), weighted-loss training with early stopping, evaluation.
-   `scripts/gbt_baseline.py` -- XGBoost on 55 aggregated per-event
    features. Beats all three deep models.
-   `scripts/features.py` -- 198 per-event features: distributional
    statistics plus physics-derived ones, notably **PDOF** (principal
    direction of force, the standard forensic descriptor of impact
    direction) and lateral-vs-longitudinal energy ratios.
-   `scripts/ensemble.py` -- **the current best result.** Soft-voting
    ensemble, evaluated with repeated cross-validation.
-   `scripts/cross_validate.py`, `ablations.py`,
    `random_split_diagnostic.py` -- single-pass CV and secondary
    experiments.

## Results

All numbers below test on **real CISS crashes only**. Macro-F1 is
computed over the 4 realizable classes -- CISS contains zero Head-on
events, so that class is in the label space (the training pool has 1,204)
but can never be a correct answer.

| Model | Protocol | Accuracy | Macro-F1 |
| --- | --- | --- | --- |
| Majority-class baseline | -- | 38.9% | -- |
| Best deep sequence model | Mixed split | 51.8% | 0.389 |
| XGBoost, 55 features | 5-fold CV | 51.7% +/- 1.7% | 0.415 +/- 0.018 |
| XGBoost, 55 features | Repeated CV (3x5) | 52.1% +/- 2.4% | 0.420 +/- 0.018 |
| **Soft-voting ensemble** | **Repeated CV (3x5)** | **53.9% +/- 2.2%** | **0.436 +/- 0.026** |

The ensemble combines three deliberately *dissimilar* members, so they
fail on different events:

1.  XGBoost on 55 aggregated features, mixed training pool.
2.  XGBoost on 198 physics features, trained on **real CISS only** --
    weakest member by accuracy (50.2%), strongest by macro-F1 (0.433),
    because it never sees the synthetic domain.
3.  ExtraTrees on 198 physics features -- bagged randomized trees, a
    different bias/variance profile from boosting.

Evaluation uses **repeated** CV: three full 5-fold passes with different
fold assignments (15 estimates). This matters because per-fold accuracy
varies by 2-3 points, so a single 5-fold pass can show a "gain" that is
pure fold luck -- during development, richer features looked like +0.7pp
on one seed and measured -0.18pp once repeated over three.

Per-class (pooled, every CISS event tested once):

| Class | Precision | Recall | F1 | n |
| --- | ---: | ---: | ---: | ---: |
| Rear-end | 0.49 | 0.52 | 0.50 | 495 |
| Angle | 0.43 | 0.29 | 0.34 | 345 |
| Single-vehicle | 0.67 | 0.78 | 0.72 | 591 |
| Sideswipe | 0.26 | 0.24 | 0.25 | 89 |

## Why ~54% and not 80%

Two measured anchors bound this task:

-   **Floor:** 38.9% (always predict the majority class).
-   **Ceiling:** ~70% accuracy / 0.71 macro-F1. This is the
    `random_split_diagnostic.py` result -- identical features and
    architecture, but train and test drawn from the *same* distribution.
    With zero domain shift, this task still tops out near 70%.

So 53.9% recovers roughly **80% of the model's own achievable ceiling**
under genuine cross-domain shift. The gap between 54% and 70% is the
sim-to-real generalization gap, quantified rather than hidden -- that
measurement is itself one of this project's contributions. The gap
between 70% and 100% is inherent: five crash types are not cleanly
separable from pre-crash EDR kinematics alone.

Sideswipe remains the weakest class (89 CISS events; only 17 in CTDB,
because standardized crash tests use canonical impact angles that rarely
produce glancing contact). The clearest remaining lever is generating
BeamNG sideswipe scenarios specifically.

## Reproducing

```
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install numpy pandas scikit-learn matplotlib seaborn joblib tqdm xgboost
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121

# best result -- rebuilds its own tensors, ~8 min
.venv\Scripts\python.exe scripts\ensemble.py

# single-pass CV baseline
.venv\Scripts\python.exe scripts\cross_validate.py

# deep sequence models (secondary)
.venv\Scripts\python.exe scripts\build_sequences.py --split-mode mixed
.venv\Scripts\python.exe scripts\train.py --split-mode mixed
.venv\Scripts\python.exe scripts\evaluate.py --split-mode mixed

# strict zero-real-crashes-in-training protocol (hardest, for comparison)
.venv\Scripts\python.exe scripts\build_sequences.py
.venv\Scripts\python.exe scripts\train.py
.venv\Scripts\python.exe scripts\evaluate.py
```

Trained on an RTX 3050 laptop GPU; the deep models converge in under 2.5
minutes each, and `ensemble.py` takes roughly 8 minutes end to end.

## Phase 3 onward: what is pending

Phase 2 is complete and is the end of this delivery. Phases 3-5 are
**not implemented**:

-   **Phase 3 -- Explainability.** SHAP attribution over the ensemble's
    features and attention-weight analysis for the sequence models. Phase
    2 already saves the artifacts this needs: per-event test predictions
    and probabilities, attention weights, and a SHAP background sample.
    The physics features in `features.py` were designed partly with this
    in mind -- "PDOF was 87 degrees, so this was a lateral impact" is a
    directly interpretable explanation in a way that "hidden unit 43
    fired" is not.
-   **Phase 4 -- Physics validation.** Cross-check predicted crash type
    against momentum/energy consistency (CISS carries `DVTOTAL`,
    `DVLONG`, `DVLAT` for exactly this).
-   **Phase 5 -- 3D reconstruction.** Reconstruct the crash in BeamNG,
    using the predicted class to select the reconstruction template.

For the intended forensic use case, the classifier's **calibrated
probabilities matter more than its argmax**. A tool of this kind is meant
to separate cases it can classify confidently from cases that genuinely
need a human expert -- a flagged ambiguous case is a useful output, not a
failure. Phase 3's explainability work is what would make that
confidence auditable.

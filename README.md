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

"""
Shared constants for the Phase 2 training pipeline.

Every other script in scripts/ imports from here so that the feature set,
sequence length, and split rule are defined in exactly one place. If you
change T or FEATURE_COLS, do it here and rerun build_sequences.py — every
downstream script (train.py, evaluate.py, ablations.py) picks it up
automatically.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = REPO_ROOT / "Dataset" / "data" / "training_dataset_final.csv"
TENSOR_DIR = REPO_ROOT / "models" / "tensors"
MODEL_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"

RANDOM_STATE = 42

# Sequence length. Chosen from the ACTUAL post-dedup per-event timestep
# distribution (see documentation.txt Part 3.3/8): after fixing the
# duplicate-timestamp bug, 90th percentile = 151, 95th = 161, max = 201.
# T=160 keeps ~95% of SynSHRP2/BeamNG training events untruncated while
# not wasting a third of every batch on padding the way T=200 would.
# (An experiment with T=50, matching CISS's much shorter ~5s pre-impact
# window, was also tried to test whether a horizon mismatch explained the
# cross-source generalization gap -- documentation.txt Part 9.4 -- but
# performed no better, and worse for the Transformer, so T=160 stands.)
SEQ_LEN = 160

# Raw signal columns that are genuinely usable across sources (see
# documentation.txt Part 3.2 for why engine_rpm / abs_esc_active /
# abs_active were dropped: they are >92% missing and, critically, 100%
# missing for BOTH training sources (SynSHRP2 and BeamNG) simultaneously
# -- a feature that is constant zero for the model's entire training set
# cannot be learned from no matter how sparse it is on the CISS test set).
RAW_FEATURE_COLS = [
    "speed_kmh", "accel_long", "accel_lat",
    "brake_active", "throttle_pct", "steering_deg", "yaw_rate",
]

# Columns that get an explicit *_missing indicator instead of a silent
# fill. throttle_missing / steering_missing already existed in Phase 1's
# output; brake_missing and yaw_rate_missing are computed fresh in
# build_sequences.py because Phase 1 didn't generate them even though
# both fields are meaningfully sparse per-source (see documentation.txt
# Part 3.2 decision log).
INDICATOR_FOR = ["brake_active", "throttle_pct", "steering_deg", "yaw_rate"]

FEATURE_COLS = RAW_FEATURE_COLS + [f"{c}_missing" for c in INDICATOR_FOR]
# Final order (11 columns), fixed and saved to feature_cols.json so
# evaluate.py / ablations.py / any later Phase 3 code never has to guess it:
#   speed_kmh, accel_long, accel_lat, brake_active, throttle_pct,
#   steering_deg, yaw_rate, brake_active_missing, throttle_pct_missing,
#   steering_deg_missing, yaw_rate_missing

# Physically-motivated sanity bounds used to sanitize raw sensor values
# before anything else happens. Discovered necessary during Phase 2: the
# Phase 1 output contains un-decoded NHTSA "unknown" sentinel codes
# (repeating values like 99996.0 / 99997.0 -- a giveaway of a raw coded
# field that was never converted to physical units or NaN) and other
# corrupted readings, present in EVERY source but far more severe in
# CISS. A handful of examples that make the case on their own: CISS
# accel_long reaches 425,115 m/s^2 (43,000g); SynSHRP2 steering_deg
# reaches 81,779 degrees (227 full lock-to-lock turns); CISS brake_active
# -- documented in CISS_DATA_CODES_GUIDE.md as a Binary field -- takes
# values up to 131,070. Left unfixed, StandardScaler (fit on the train
# split) turns these into z-scores in the thousands and the model
# collapses on the CISS test set (see documentation.txt Part 9 for the
# full before/after). Bounds below are deliberately generous -- wide
# enough to keep any real extreme crash reading, tight enough to catch
# anything that is obviously a leftover numeric code:
PHYSICAL_BOUNDS = {
    "speed_kmh": (0.0, 250.0),        # highway + margin; passenger car
    "accel_long": (-200.0, 200.0),    # ~20g, above real EDR impact peaks
    "accel_lat": (-200.0, 200.0),
    "throttle_pct": (0.0, 100.0),     # sensor-defined range
    "steering_deg": (-1080.0, 1080.0),  # 3 full lock-to-lock turns
    "yaw_rate": (-500.0, 500.0),      # deg/s; near one rotation/sec
    # brake_active is handled separately: it must be exactly {0.0, 1.0}
    # (see CISS_DATA_CODES_GUIDE.md UIR mapping table), any other value
    # is a corrupted/un-decoded reading, not an out-of-range physical one.
}

CLASS_NAMES = ["Rear-end", "Angle", "Head-on", "Single-vehicle", "Sideswipe"]
NUM_CLASSES = 5

TRAIN_SOURCES = ("SHRP2", "BeamNG")
TEST_SOURCE = "CISS"

# README's "Final Sample Weight = Source Weight x Class Weight" -- SynSHRP2
# is naturalistic real-driving data, BeamNG is fully synthetic and
# discounted slightly. CISS is real crash data (used only in --split-mode
# mixed, where a fraction of it enters the train pool -- see
# CISS_TRAIN_FRACTION below) and is weighted highest since it's the actual
# target domain the model is ultimately evaluated against.
SOURCE_WEIGHTS = {"SHRP2": 1.0, "BeamNG": 0.8, "CISS": 1.2}

# --split-mode mixed (documentation.txt Part 12): fraction of CISS events
# (stratified by crash_class) that enters the train/val pool alongside
# SynSHRP2+BeamNG; the remaining (1 - CISS_TRAIN_FRACTION) stays held out
# as the real-world test set. 0.7 was chosen to leave a meaningful
# held-out real-crash test set (~30% of 1,520 = ~456 events) while giving
# the model enough real examples per class to actually learn from --
# Sideswipe only has 89 CISS events total, so even at 0.7 that's just
# ~62 real training examples for the rarest class.
CISS_TRAIN_FRACTION = 0.7

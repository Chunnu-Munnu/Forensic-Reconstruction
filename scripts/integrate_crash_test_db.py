"""
Converts the downloaded NHTSA crash-test accelerometer curves
(Dataset/data/raw/CrashTestDB/) into UIR-schema rows and appends them to
training_dataset_final.csv as a 5th data source ("CTDB").

Design decisions (documented, not silent):

1. Axis mapping: prefers LOCAL-frame curves (X-LOCAL=accel_long,
   Y-LOCAL=accel_lat -- already vehicle-relative, same convention as CISS)
   when available for a test; falls back to GLOBAL-frame (X-GLOBAL,
   Y-GLOBAL) otherwise, which is an approximation (assumes the vehicle's
   heading was roughly aligned with the lab's global X axis at impact --
   true for most NCAP head-on/rear configurations, weaker for angled
   IMPACTOR INTO VEHICLE tests). ~68% of downloaded curves are GLOBAL-only.

2. Units: accel curves are in G's -> multiplied by 9.81 for m/s^2, same
   as every other source. closingSpeed from the test-index metadata has
   no explicit unit field in the API; ASSUMED to already be km/h (values
   like 64.2 for a 1979 NCAP frontal test are physically consistent with
   a ~40 mph target speed in km/h, and this project's UIR convention is
   km/h throughout) -- flagged here as an assumption, not verified.

3. Speed is not directly recorded (these are accelerometers, not speed
   sensors). Derived by backward integration from the one known anchor
   point: speed at the recorded window's t=0 (impact instant) equals
   closingSpeed, then speed(t) = closingSpeed - integral_t^0 accel_long,
   using cumulative trapezoidal integration over the RAW high-rate curve
   before resampling. Clipped to >=0 (same physical prior as
   build_sequences.py's negative-speed handling for other sources).

4. brake_active / throttle_pct / steering_deg / yaw_rate are NOT recorded
   by these tests (no live driver telemetry in a staged crash test) --
   set to NaN with their *_missing indicator = 1, same treatment as any
   other source missing a field.

5. Resampling: each test's raw ~20kHz, ~0.35s curve is resampled to 50
   evenly-spaced points across its own real duration (NOT forced onto a
   sparse 10Hz/16s grid like the other sources use) -- otherwise a 0.35s
   pulse would collapse to 3-4 real timesteps out of build_sequences.py's
   T=160, discarding almost all of the actual crash-pulse shape. This is
   fine: build_sequences.py only needs real (event_id, t) pairs in
   seconds: it doesn't require a fixed sample rate across sources.

Usage:
    .venv\\Scripts\\python.exe scripts\\integrate_crash_test_db.py
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "Dataset" / "data" / "raw" / "CrashTestDB" / "manifest.csv"
LIVE_INDEX = REPO_ROOT / "Dataset" / "data" / "raw" / "CrashTestDB" / "test_metadata_index_live.csv"
FINAL_CSV = REPO_ROOT / "Dataset" / "data" / "training_dataset_final.csv"
BACKUP_CSV = REPO_ROOT / "Dataset" / "data" / "training_dataset_final.backup_pre_ctdb.csv"

N_SAMPLES = 50
G = 9.81

UIR_COLS = ["PTIME", "speed_kmh", "accel_long", "accel_lat", "throttle_pct", "steering_deg",
            "engine_rpm", "brake_active", "abs_esc_active", "CASEID", "VEHNO", "crash_class",
            "CRASHTYPE", "DVTOTAL", "DVLONG", "DVLAT", "SURFCOND", "LIGHTCOND", "WEATHER",
            "MODELYR", "source_year", "event_id", "source", "is_crash", "t", "yaw_rate",
            "throttle_missing", "steering_missing"]


def load_curve(path):
    t, v = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            try:
                t_val, v_val = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            t.append(t_val)
            v.append(v_val)
    return np.array(t), np.array(v)


# Raw crash-test accelerometer channels have known, expected high-frequency
# sensor noise/resonance spikes (a single sample briefly reading 200-300G
# is a real, documented artifact of unfiltered crash-test data -- SAE J211
# is the standard that defines low-pass "CFC" filtering for exactly this
# reason). Left unfiltered, a single spike poisons this script's cumulative
# speed integration for the ENTIRE event (one bad instant creates a
# permanent velocity offset for every timestep after it). Two fixes,
# applied in this order, before integration ever runs:
#   1. Clip to +-150G (~1471 m/s^2) -- generous enough to keep any real
#      severe crash-pulse peak, tight enough to reject spike artifacts.
#   2. A short moving-average (5 raw samples, ~0.25ms at 20kHz) to smooth
#      remaining high-frequency noise -- a crude stand-in for a proper
#      SAE J211 CFC low-pass filter, good enough for this purpose.
CLIP_G = 150.0


def clean_curve(v_g):
    v_clipped = np.clip(v_g, -CLIP_G, CLIP_G)
    kernel = np.ones(5) / 5
    return np.convolve(v_clipped, kernel, mode="same")


def pick_axis_curves(rows_for_test):
    """rows_for_test: list of manifest rows (dicts) for one testNo.
    Returns (x_path, y_path) preferring LOCAL over GLOBAL, or (None, None)."""
    by_axis = {}
    for r in rows_for_test:
        axis = r["axis"]
        by_axis.setdefault(axis, r["file_path"])
    x = by_axis.get("X - LOCAL") or by_axis.get("X - GLOBAL")
    y = by_axis.get("Y - LOCAL") or by_axis.get("Y - GLOBAL")
    return x, y


def main():
    # --- load manifest, group by testNo ---
    by_test = {}
    with open(MANIFEST, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["curveNo"]:
                continue
            by_test.setdefault(row["testNo"], []).append(row)

    print(f"Tests with curve data: {len(by_test)}")

    # --- load closingSpeed / testDate metadata ---
    meta = {}
    with open(LIVE_INDEX, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            meta[row["testNo"]] = row

    new_rows = []
    skipped_no_speed, skipped_no_axes, n_local, n_global = 0, 0, 0, 0

    for test_no, rows in by_test.items():
        x_path, y_path = pick_axis_curves(rows)
        if x_path is None or y_path is None:
            skipped_no_axes += 1
            continue
        if any(r["axis"] == "X - LOCAL" for r in rows):
            n_local += 1
        else:
            n_global += 1

        m = meta.get(test_no)
        if not m or not m.get("closingSpeed"):
            skipped_no_speed += 1
            continue
        try:
            closing_speed_kmh = float(m["closingSpeed"])
        except ValueError:
            skipped_no_speed += 1
            continue

        t_x, ax_g = load_curve(REPO_ROOT / x_path)
        t_y, ay_g = load_curve(REPO_ROOT / y_path)
        if len(t_x) < 5 or len(t_y) < 5:
            continue
        ax_g, ay_g = clean_curve(ax_g), clean_curve(ay_g)

        order = np.argsort(t_x)
        t_sorted, a_sorted = t_x[order], (ax_g * G)[order]

        y_order = np.argsort(t_y)
        t_y_sorted, ay_sorted = t_y[y_order], (ay_g * G)[y_order]
        accel_lat_on_x_grid = np.interp(t_sorted, t_y_sorted, ay_sorted)

        # backward integration from t=0 (impact) using the RAW high-rate series
        # speed(t) = closing_speed - integral_t^0 accel_long(s) ds
        cum = np.concatenate([[0.0], np.cumsum(
            (a_sorted[1:] + a_sorted[:-1]) / 2 * np.diff(t_sorted))])  # trapezoidal, m/s
        speed_at_t0 = closing_speed_kmh / 3.6  # to m/s
        idx0 = np.searchsorted(t_sorted, 0.0)
        cum_at_0 = cum[min(idx0, len(cum) - 1)]
        speed_mps = speed_at_t0 - (cum_at_0 - cum)  # m/s, full series
        speed_kmh_series = np.clip(speed_mps * 3.6, 0, None)

        # resample to N_SAMPLES evenly spaced points across the real duration
        t_min, t_max = t_sorted.min(), t_sorted.max()
        t_grid = np.linspace(t_min, t_max, N_SAMPLES)
        accel_long_rs = np.clip(np.interp(t_grid, t_sorted, a_sorted), -200.0, 200.0)
        accel_lat_rs = np.clip(np.interp(t_grid, t_sorted, accel_lat_on_x_grid), -200.0, 200.0)
        speed_rs = np.clip(np.interp(t_grid, t_sorted, speed_kmh_series), 0.0, 250.0)

        cls = int(float(rows[0]["crash_class"]))
        try:
            year = int(m["testDate"][:4]) if m.get("testDate") else 2000
        except ValueError:
            year = 2000

        for i in range(N_SAMPLES):
            new_rows.append({
                "PTIME": np.nan, "speed_kmh": speed_rs[i], "accel_long": accel_long_rs[i],
                "accel_lat": accel_lat_rs[i], "throttle_pct": np.nan, "steering_deg": np.nan,
                "engine_rpm": np.nan, "brake_active": np.nan, "abs_esc_active": np.nan,
                "CASEID": np.nan, "VEHNO": np.nan, "crash_class": cls, "CRASHTYPE": np.nan,
                "DVTOTAL": np.nan, "DVLONG": np.nan, "DVLAT": np.nan, "SURFCOND": np.nan,
                "LIGHTCOND": np.nan, "WEATHER": np.nan, "MODELYR": np.nan,
                "source_year": year, "event_id": f"CTDB_{test_no}", "source": "CTDB",
                "is_crash": 1, "t": t_grid[i], "yaw_rate": np.nan,
                "throttle_missing": 1, "steering_missing": 1,
            })

    print(f"Skipped (no axis pair): {skipped_no_axes}")
    print(f"Skipped (no closing speed): {skipped_no_speed}")
    print(f"Used LOCAL frame: {n_local}, GLOBAL frame (fallback): {n_global}")
    n_events = len(new_rows) // N_SAMPLES
    print(f"New events: {n_events}, new rows: {len(new_rows)}")

    new_df = pd.DataFrame(new_rows)[UIR_COLS]

    print(f"\nBacking up original to {BACKUP_CSV.name}")
    if not BACKUP_CSV.exists():
        Path(FINAL_CSV).replace(BACKUP_CSV) if False else None  # never move; copy instead
    import shutil
    if not BACKUP_CSV.exists():
        shutil.copy(FINAL_CSV, BACKUP_CSV)

    existing = pd.read_csv(FINAL_CSV, low_memory=False)
    print(f"Existing rows: {len(existing)}, existing events: {existing['event_id'].nunique()}")

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(FINAL_CSV, index=False)
    print(f"Combined rows: {len(combined)}, combined events: {combined['event_id'].nunique()}")
    print("Class distribution of new CTDB events:")
    print(new_df.drop_duplicates("event_id")["crash_class"].value_counts().sort_index())


if __name__ == "__main__":
    main()

# SynSHRP2 Data Cleaning: Method and Rationale

## What is SynSHRP2?

SynSHRP2 is a publicly available synthetic dataset derived from the SHRP2 Naturalistic Driving Study. It contains 6,531 events (crashes and near-crashes) with kinematic time-series data (speed, acceleration, braking) recorded at 100ms intervals. Unlike CISS (which records only the final 5-20 seconds before impact), SynSHRP2 captures the full pre-crash sequence — sometimes 15-30 seconds of driving behavior leading up to the critical event.

---

## Why We Need SynSHRP2

CISS alone gives us ~3,500-4,500 clean crash events after filtering. That's too small for robust Transformer training, especially for rare crash types (head-on: ~5%, single-vehicle: ~10%). SynSHRP2 adds:

1. **Volume:** 6,531 additional events (1,340 actual crashes + 5,191 near-crashes)
2. **Long pre-crash windows:** Full driving context, not just final seconds
3. **Near-crash data:** Events that looked dangerous but didn't result in collision — critical for the model to learn what "almost but didn't" looks like
4. **Diversity:** Different driving scenarios, speeds, and maneuvers

---

## Cleaning Steps Explained

### Step 1: Load Metadata (CSV) + Map Incident Types to Crash Classes

**What we did:**
- Loaded `Tabular_records.csv` containing 6,531 events
- Each row has: Event_ID, Event_type (Crash vs Near-Crash), Incident_type (rear-end, sideswipe, etc.)
- Mapped `Incident_type` strings to our 5 crash classes (0-4)

**Why:**
- Incident_type is categorical text ("Rear-end, striking", "Road departure (left or right)")
- Our model needs numerical labels (0-4)
- Unmapped types (n=828, e.g., "Animal-related", "Other") were dropped because they don't fit our 5-class scheme
- Result: 5,703 usable events (87.3% of original)

**Mapping:**
```
Rear-end (striking/struck) → 0
Turn into/across path, intersection → 1 (Angle crash)
Head-on, opposite direction → 2
Road departure, fixed object, backing → 3 (Single-vehicle)
Sideswipe (same/opposite direction) → 4
Animal-related, Other → Dropped
```

### Step 2: Load JSON Kinematic Data (One File Per Event)

**What we did:**
- Each event has a separate JSON file in `Kinematic_Signals/` folder
- Each JSON is an array of ~2,600 timesteps with sensor readings
- Timestep format: `{"TimeStamp": -9800, "Lon_Acc": 0.0, "Lat_Acc": -0.006, "Speed": null, ...}`

**Why JSON instead of one CSV:**
- SynSHRP2 organizes data per-event for easier loading/management
- We read each JSON, convert to a dataframe, apply UIR schema

### Step 3: Normalize Timestamps Relative to Impact

**What we did:**
```python
t = (TimeStamp - Impact) / 1000.0
```
- `TimeStamp` is in milliseconds
- `Impact` is the recorded collision moment (from metadata)
- Result: t in seconds, with t=0 at collision, negative t = pre-crash

**Why:**
- SynSHRP2 uses absolute timestamps (e.g., -9800ms = 9.8 seconds before impact)
- Our UIR standard requires t=0 at collision for consistency with CISS
- Allows us to always trim to -20s to 0s window (pre-crash period only)

**Example:**
- TimeStamp = -8000ms, Impact = 4200ms
- t = (-8000 - 4200) / 1000 = -12.2 seconds (12.2 seconds before collision)

### Step 4: Filter to Pre-Crash Window (-20s to 0s)

**What we did:**
```python
df = df[(df['t'] >= -20) & (df['t'] <= 0)]
```

**Why:**
- CISS EDR data is 5-20 seconds pre-crash
- We standardize SynSHRP2 to the same window for consistency
- Events with <5 timesteps in this window are dropped (not enough data)
- Ensures Transformer training sees similar-length sequences from both sources

### Step 5: Unit Conversions

**Acceleration: g to m/s²**
```python
accel_long = Lon_Acc * 9.81
accel_lat = Lat_Acc * 9.81
```
Why: SynSHRP2 records acceleration in gravitational units (g); we convert to SI units (m/s²) to match CISS and physics calculations.

**Speed: m/s to km/h**
```python
speed_kmh = Speed * 3.6
```
Why: SynSHRP2 records speed in meters per second; we convert to km/h to match CISS convention and real-world crash speed literature.

### Step 6: Map Missing Fields to NaN

**What we did:**
SynSHRP2 does NOT record:
- Throttle percentage → `throttle_pct = NaN`, `throttle_missing = 1`
- Steering angle → `steering_deg = NaN`, `steering_missing = 1`
- Engine RPM → `engine_rpm = NaN`
- ABS/ESC status → `abs_esc_active = NaN`

**Why:**
- SynSHRP2 is from naturalistic driving study; only has core kinematic sensors
- CISS has these fields because it's from CDR (crash data recorder) designed for detailed forensics
- Marking as NaN + missing indicator tells the model: "this field wasn't recorded for this source"
- Model learns to weight available fields more heavily when optional fields are absent

**Brake status special case:**
- SynSHRP2 has `Ped_BS` (pedal brake status): 1.0 = brake pressed, 0 = not pressed
- CISS has `BRKSTAT`: binary 0/1
- We use `Ped_BS` directly as `brake_active`

### Step 7: Add Source Tags and Event Labels

**What we did:**
```python
event_id = f"SHRP2_{original_event_id}"
crash_class = mapped from Incident_type
is_crash = 1 if Event_type == 'Crash' else 0
source = 'SHRP2'
```

**Why:**
- Unique event IDs prevent collisions with CISS case IDs
- `crash_class` is the training label (what we predict)
- `is_crash` distinguishes actual crashes (n=1,340) from near-crashes (n=5,191)
  - Near-crashes are valuable negative examples: driving that looked risky but didn't crash
  - Helps model distinguish "dangerous but avoided" from "dangerous and crashed"
- `source` tag allows us to track data provenance and apply per-source weighting during training

### Step 8: Build UIR (Unified Intermediate Representation)

**What we did:**
Selected exactly these columns for every timestep from every event:
```
event_id, source, source_year, crash_class, is_crash,
t, speed_kmh, accel_long, accel_lat, brake_active,
throttle_pct, steering_deg, yaw_rate, engine_rpm, abs_esc_active,
throttle_missing, steering_missing
```

**Why:**
- Uniform schema across CISS (different source) and SynSHRP2
- Every row follows same format → Transformer sees consistent input
- Missing indicators allow model to handle heterogeneous data sources
- Easy to merge with CISS data later: concatenate and the schemas align perfectly

### Step 9: Quality Checks and Class Distribution

**Checks performed:**
- Speed range: confirm km/h conversion looks reasonable
- Acceleration ranges: confirm g→m/s² conversion correct
- Missing values: count NaNs to assess data completeness
- Class distribution: ensure all 5 crash types are represented
- Crash vs near-crash split: verify we have both

**Why:**
- Detects unit conversion errors early
- Identifies data quality issues (e.g., if 50% of speeds are NaN)
- Class imbalance indicates whether we need oversampling
- Near-crash ratio tells us if model will see enough negative examples

---

## Key Differences: CISS vs SynSHRP2

| Aspect | CISS | SynSHRP2 |
|---|---|---|
| Pre-crash window | 5-20 seconds | Full drive (up to 30+ seconds) |
| Source | Real EDR data from actual crashes | Synthetic kinematic data from naturalistic study |
| Event type | Crashes only | Crashes + near-crashes |
| Throttle/Steering | Available | Not recorded (NaN) |
| Sampling rate | 2 Hz (interpolated to 10 Hz) | 10 Hz (100ms intervals) |
| Number of events | ~3,500-4,500 after filtering | ~5,700 after mapping |
| Use in pipeline | Test set (ground truth validation) | Training set (model learning) |

---

## Why This Approach Works

1. **Complementary strengths:** CISS is authoritative (real crashes, legally grounded); SynSHRP2 is voluminous (training data, behavioral context)

2. **Consistent schema:** UIR normalization means the Transformer sees uniform inputs regardless of source

3. **Missing data handling:** Missing indicators let the model learn which features are reliable vs. optional per source

4. **Balanced training:** Near-crashes teach the model "what almost went wrong" without actually crashing — critical for nuanced crash prediction

5. **Generalization:** Training on two different data sources (CISS + SynSHRP2) prevents overfitting to one data collection method

---

## What Happens Next

After cleaning both CISS and SynSHRP2:
1. Concatenate into single unified dataset
2. Apply source weighting (CISS overweighted despite smaller size, because it's real crashes)
3. Split: CISS-only as test set; SynSHRP2 + synthetic BeamNG as training set
4. Train Transformer on training set
5. Evaluate on held-out CISS test set (real crashes)
6. Report performance metrics per source, per crash class, per data quality tier

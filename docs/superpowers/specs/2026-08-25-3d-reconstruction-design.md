# Phase 5 — 3D Reconstruction via BeamNG Template Replay

**Date:** 2026-08-25
**Status:** Approved design, self-reviewed. Awaiting implementation.

---

## 0. Context

Phase 5 of the project — 3D reconstruction — is currently described in
`docs/documentation.txt` Part 8.0 as a one-line outline ("renders the
validated reconstruction as a multi-angle video and examinable 3D scene")
and is explicitly out of scope for the Phase 2 delivery. This spec
designs the first concrete implementation of Phase 5, scoped to
**template replay** with parametric variation driven by per-event
classifiers and extracted physics features.

The classifier from Phase 2 (`scripts/ensemble.py`, 53.9% ± 2.2%
accuracy on real CISS crashes) produces a predicted crash class for
each event. The physics features in `scripts/features.py` already
extract **PDOF** (principal direction of force — the standard forensic
descriptor of impact direction), peak accelerations, max yaw rate, and
speed envelopes. This spec turns those outputs into a BeamNG simulation
that visually demonstrates the crash.

The existing `Dataset/data/beamNG.py` is the data generator that
produces kinematic training CSVs for the classifier. **Per explicit
user direction, it must not be modified.** All new code lives under
`scripts/` and is a separate, self-contained subsystem.

---

## 1. Goals and Non-Goals

### In scope

1. A standalone CLI `scripts/reconstruct.py` that takes one or more
   `event_id`s and produces a BeamNG video + per-event metadata JSON.
2. A helper module `scripts/reconstruct_params.py` that converts a
   single event's classifier output + extracted scalars into concrete
   scenario parameters (approach angles, speeds, vehicle offsets,
   geometry class).
3. A scenario module `scripts/reconstruct_scenarios.py` that
   reimplements the four crash-class geometries (head-on, angle,
   single-vehicle, sideswipe) without importing `beamNG.py`.
4. A capture module `scripts/reconstruct_capture.py` that handles
   BeamNG connection lifecycle, runs the simulation, captures frames
   from two camera angles, and encodes the output video.
5. A precomputed-features cache: when reconstructing many events,
   compute the per-event scalars (PDOF, initial speed, peak
   longitudinal accel, max yaw rate) once and reuse.
6. A `predictions.csv` output from `scripts/ensemble.py` so the
   reconstruction pipeline has a stable input format.

### Explicitly NOT in scope

1. **Trajectory-following mode** — the ego vehicle does not replay the
   actual recorded steering/throttle trajectory. Scope is "class +
   parametric variation," not kinematic replay.
2. **Momentum / energy validation** — that's Phase 4
   (`docs/documentation.txt` Part 8.0). CISS's `DVTOTAL`/`DVLONG`/`DVLAT`
   are extracted but not yet consumed for validation.
3. **SHAP / per-feature explanations** — that's Phase 3. Out of scope.
4. **Multi-vehicle scenarios beyond the existing four crash classes** —
   no chain-reaction, no pedestrian strike, no motorcycle. The four
   classes the classifier outputs are what get visualized.
5. **Modifying `Dataset/data/beamNG.py`** — the existing data generator
   is untouched. All new code lives under `scripts/`.
6. **Headless / no-BeamNG fallback** — this design explicitly requires
   BeamNG.tech running locally. If BeamNG isn't installed, the CLI
   fails fast with a clear error.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  scripts/reconstruct.py   (CLI entry point)                  │
│  ──────────────────────                                     │
│  Usage:                                                     │
│    python scripts/reconstruct.py --event-id CISS_2019_…      │
│    python scripts/reconstruct.py --from-predictions \        │
│            results/ensemble/predictions.csv                 │
│    python scripts/reconstruct.py --smoke-test                │
└────────┬───────────────────────┬──────────────────┬─────────┘
         │                       │                  │
         ▼                       ▼                  ▼
┌─────────────────┐  ┌────────────────────┐  ┌──────────────────┐
│ scripts/        │  │ scripts/           │  │ scripts/         │
│ reconstruct_    │  │ reconstruct_       │  │ reconstruct_     │
│ params.py       │  │ scenarios.py       │  │ capture.py       │
│                 │  │                    │  │                  │
│ event → dict of │  │ dict → BeamNG      │  │ run sim → write  │
│ scenario params │  │ scenario object    │  │ frames + video   │
│                 │  │                    │  │                  │
│ Sources:        │  │ 4 templates:       │  │ FreeCamera sensor│
│ - features.py   │  │ - head-on          │  │ + manual frame   │
│   (PDOF, speed, │  │ - angle            │  │ capture at       │
│   accel peaks)  │  │ - single-vehicle   │  │ 10 Hz            │
│ - ensemble.py   │  │ - sideswipe        │  │                  │
│   predictions   │  │                    │  │ Output:          │
│   CSV output    │  │ Uses SAME map      │  │ results/         │
│ - raw CSV for   │  │ (smallgrid),       │  │ reconstruction/  │
│   scalars not   │  │ vehicle model      │  │ <event_id>/      │
│   in features   │  │ (etk800), 60 Hz    │  │  ├── frame_NNN   │
│                 │  │ physics as the     │  │  ├── metadata    │
│                 │  │ generator          │  │  │   .json        │
│                 │  │                    │  │  └── video.mp4   │
└─────────────────┘  └────────────────────┘  └──────────────────┘
```

### Component responsibilities

- **`reconstruct_params.py`** — pure functions over dicts. Maps
  `event_id` + `prediction_row` + `extracted_scalars` to a
  `ScenarioParams` namedtuple. **Unit-testable without BeamNG.**
- **`reconstruct_scenarios.py`** — the only file that touches
  BeamNGpy. Builds `Scenario` and `Vehicle` objects from
  `ScenarioParams`. Owns the calibration logic.
- **`reconstruct_capture.py`** — owns the BeamNG connection
  lifecycle, runs scenarios, captures frames, encodes videos.
  Independent of the physics scenario.
- **`reconstruct.py`** — glue. Parses CLI args, orchestrates the
  pipeline, handles per-event error recovery.

---

## 3. Data Flow

For one `event_id`:

1. **Input.** CLI passes `event_id` to `reconstruct.py`.
2. **Load prediction.** Read `results/ensemble/predictions.csv` to
   get predicted class, class probabilities, and predicted PDOF bin
   for this event.
3. **Extract scalars.** Run `scripts/features.py::build_rich` on that
   event's UIR rows to get the 198 features, then pull:
   - `pdof_deg`, `pdof_abs_deg`
   - `speed_kmh_first`, `speed_kmh_max`
   - `accel_long_peak`, `accel_lat_peak`
   - `yaw_rate_absmax`
   For CISS events only, additionally pull `DVTOTAL`, `DVLONG`,
   `DVLAT` from the raw CSV.
4. **Map to params.** `reconstruct_params.event_to_scenario_params(
   event_id, prediction_row, scalars)` returns a `ScenarioParams`
   dict or `None` if the event should be skipped.
5. **Build scenario.** `reconstruct_scenarios.build_replay_scenario(
   params, qh)` returns a configured `Scenario` with two vehicles
   and attached FreeCamera sensors.
6. **Run + capture.** `reconstruct_capture.run_and_capture(scenario,
   ego, params, output_dir)` starts BeamNG, runs physics, captures
   frames at impact and ~0.6s post-impact, writes them to
   `output_dir`.
7. **Encode + metadata.** `reconstruct_capture.assemble_video(
   frames, event_id, output_dir)` writes `video.mp4` (mp4v codec via
   OpenCV) and `metadata.json` with predicted class, probability
   distribution, observed vs reconstructed scalars, and frame
   timestamps.

### Output layout

```
results/reconstruction/<event_id>/
    metadata.json       # prediction, params, status, frame timestamps
    frame_overview_NNN.png   # from stationary FreeCamera
    frame_chase_NNN.png      # from chase FreeCamera
    video_overview.mp4
    video_chase.mp4
```

---

## 4. Parametric Variation per Crash Class

### 4.1 Head-on (class 2)

- **Canonical geometry:** two vehicles on the same line, opposite
  headings, equal approach distance.
- **Varied by:** ego speed, target speed (within ±15% of observed
  ego `speed_kmh_max`).
- **Fixed:** collinear placement (PDOF for head-on is 0° / 180° ± 10°).
- **Visual goal:** viewer sees two vehicles closing head-on at the
  recorded speeds.

### 4.2 Angle / intersection (class 1)

- **Canonical geometry:** ego drives +X, target approaches from +Y
  at 90° (T-bone into ego's left side).
- **Varied by:**
  - **PDOF angle** — `pdof_abs_deg` shifts the target's approach
    heading by `(pdof_abs_deg − 90)` so a 75° PDOF event gets a 75°
    target approach. Bounded ±25° from canonical.
  - **Lateral offset** — where on ego's side the target strikes,
    ±1.5 m.
  - **Ego speed** — from observed `speed_kmh_max`.
  - **Target speed** — half of ego speed (forensic default for
    intersection crashes; overridable from CISS DVLAT if available).
- **Visual goal:** T-bone or angled side-strike at the recorded
  impact direction.

### 4.3 Single-vehicle (class 3)

- **Canonical geometry:** ego drives +X into a static wall
  (procedural cube).
- **Varied by:** ego speed (observed `speed_kmh_max`), wall offset
  (where the obstacle sits in ego's path).
- **Visual goal:** ego strikes a fixed obstacle; outcome (deflection,
  rotation) determined by BeamNG physics.

### 4.4 Sideswipe (class 4)

- **Canonical geometry:** two vehicles abreast (ego in right lane,
  target in left lane), same speed, ego angled 3° toward target.
- **Varied by:**
  - **PDOF angle** — `(pdof_abs_deg − 90)` shifts target heading,
    but sideswipe PDOF stays near ±90° (±25° bounds).
  - **Lateral offset between vehicles** — ±0.8 m (determines
    glancing vs direct contact).
  - **Equal speed** — set to ego `speed_kmh_max`.
- **Visual goal:** two vehicles driving abreast, ego strikes target's
  flank at a glancing angle.

### 4.5 Bounds table

| Param             | Class    | Min          | Max          |
|-------------------|----------|--------------|--------------|
| ego speed         | all      | 30 km/h      | 130 km/h     |
| target speed      | 1, 2     | 0 km/h       | 130 km/h     |
| target speed      | 4        | ego − 5 km/h | ego + 5 km/h |
| PDOF offset       | 1        | −25°         | +25°         |
| PDOF offset       | 2        | −10°         | +10°         |
| PDOF offset       | 4        | −25°         | +25°         |
| lateral offset    | 1        | −1.5 m       | +1.5 m       |
| lateral offset    | 4        | −0.8 m       | +0.8 m       |

If observed values fall outside these bounds, params get clamped and
a warning is logged in `metadata.json` under `param_overrides`.

### 4.6 PDOF → heading mapping

PDOF in degrees is the angle of the acceleration vector at peak
resultant acceleration. Forensic convention:

- 0° = pure longitudinal (rear-end or head-on impact along vehicle's
  forward axis).
- ±90° = pure lateral (T-bone, sideswipe).
- intermediate = oblique.

The mapping `pdof_abs_deg → target_heading_offset` is class-dependent:

- Class 2 (head-on): `target_heading = 180°` (head-on) regardless of
  PDOF.
- Class 1 (angle): `target_heading = pdof_abs_deg`.
- Class 4 (sideswipe): `target_heading = pdof_abs_deg` with sideswipe
  tighter bounds.

### 4.7 Class 0 (rear-end) handling

Per the existing data generator's own comment
(`Dataset/data/beamNG.py` lines 58-63), rear-end is intentionally
not synthesized because it requires a closing-geometry with a
lead vehicle. Reconstruction skips class-0 events with a warning
logged to `metadata.json` under `status: "skipped_class_0"`. This
is a known limitation, documented in the CLI `--help` output.

---

## 5. BeamNG Integration and Capture

### 5.1 Connection

`reconstruct_capture.py` opens its own `BeamNGpy(HOST, PORT,
launch=True)` instance at the start of a CLI invocation. We do not
share the connection with the data generator — data generation is a
long overnight run, reconstruction is one-off.

- Single shared connection across all events in one CLI invocation.
- `bng.close()` in a `finally` block so a keyboard interrupt or crash
  mid-event doesn't leave BeamNG running.
- Connection host/port/map config: same `localhost:64256` and
  `smallgrid` map as the data generator, but configurable via CLI
  flags (`--host`, `--port`, `--map`).

### 5.2 Calibration

`reconstruct_scenarios.py::calibrate(bng)` reimplements the
data-generator's probe-vehicle logic:

1. Spawn probe at origin facing 0°, measure observed heading.
2. Repeat at +90° command, derive convention sign `s`.
3. Return a `qh(theta_deg)` closure.

Same logic, separate code, no import of `beamNG.py`. Calibration
runs once per CLI invocation.

### 5.3 Per-scenario timeout

12 seconds of simulated time per event (matching the data generator).
If no impact is detected within 12s, fail that event with a clear
error in `metadata.json` (`status: "timeout"`) and continue to the
next.

### 5.4 Capture

For visualization we add two sensors to the ego vehicle (the data
generator attaches only `Electrics` and `Damage`):

1. **`FreeCamera` (overview)** — stationary at `(0, 50, 30)` looking
   at origin, FOV 60°. Captures a wide view of the impact zone.
2. **`FreeCamera` (chase)** — follows ego at offset `(0, -8, 4)`,
   FOV 75°. Captures driver-perspective view.

Frames saved as PNGs at 10 Hz (every 6 physics steps at 60 Hz).
Two cameras × 10 Hz × up to ~120 frames = 240 images per event max.

### 5.5 Video encoding

OpenCV's `VideoWriter` with `mp4v` codec. No ffmpeg dependency.
Resolution 1280×720. Frame rate matches the capture rate.

`opencv-python` is one new dependency (not currently in `.venv`).

---

## 6. Error Handling

| Failure                              | Detection                       | Behavior                                                                                         |
|--------------------------------------|---------------------------------|--------------------------------------------------------------------------------------------------|
| BeamNG not installed / not running   | `BeamNGpy.open()` raises        | Fail fast, log to stderr, exit code 2. No `metadata.json` written.                               |
| BeamNG crashes mid-scenario          | Exception in physics loop       | Close scenario, log error, write `metadata.json` with `status: "beamng_error"`, continue.        |
| Scenario times out (no impact)        | 12 s elapsed without damage     | `status: "timeout"` in `metadata.json`. Continue.                                                 |
| Event ID not in predictions CSV      | File lookup miss                | Fail fast with list of available IDs, exit code 3.                                               |
| Event has insufficient features      | `features.py` returns NaN keys  | Skip event, `status: "insufficient_data"`, log warning.                                          |
| Predicted class is `0` (rear-end)    | Per `beamNG.py` lines 58-63     | Skip event, `status: "skipped_class_0"`. Known limitation.                                       |
| Disk space < 500 MB                  | Pre-launch check                | Abort before starting BeamNG with clear error.                                                   |

---

## 7. New Dependency on `ensemble.py`

`reconstruct.py` needs a stable input format from the ensemble. Today
the ensemble prints to stdout and writes `results/ensemble/ensemble_report.txt`.
We will teach `ensemble.py` to additionally write:

```
results/ensemble/predictions.csv
```

with columns: `event_id, true_class, pred_class, proba_0, proba_1,
proba_2, proba_3, proba_4, fold, seed`. Pooled across all 15 folds
(3 seeds × 5 folds), with `fold` and `seed` so we can reconstruct
specific test events.

This is a small, additive change to `ensemble.py`. Does not change
any reported metrics. Output columns are stable across runs (no
timestamp in filename).

---

## 8. Testing Strategy

### 8.1 Unit tests (no BeamNG required)

`scripts/tests/test_reconstruct_params.py` — pure-function tests,
runnable headlessly:

1. **PDOF → heading mapping per class** — given a synthetic event
   with known `pdof_deg` and known predicted class, the resulting
   `target_heading_deg` is within the expected range from the bounds
   table.
2. **Speed clamping** — events with observed speed outside
   [30, 130] km/h get clamped, and `metadata.json.param_overrides`
   reflects the override.
3. **Class-0 (rear-end) skip** — predict class 0 →
   `event_to_scenario_params` returns `None`.
4. **Insufficient-data skip** — event with all-NaN `speed_kmh` →
   returns `None`, marked as `insufficient_data`.
5. **Bounds enforcement** — try to inject PDOF offset of 60° for a
   sideswipe event → output clamped to ±25°.
6. **Determinism** — same event_id + same prediction CSV → identical
   scenario params dict (sort-stable, no random state).

### 8.2 Integration test (requires BeamNG)

`scripts/tests/test_reconstruct_smoke.py` — skipped unless
`BEAMNG_HOME` env var is set:

1. **Smoke test passes** — `--smoke-test` produces `metadata.json`
   and ≥2 frames in `results/reconstruction/smoke_test/`.
2. **Single head-on event** — load `predictions.csv`, pick the first
   CISS head-on event, reconstruct, assert
   `metadata.json.status == "ok"` and video file exists with >0 bytes.
3. **Failure paths don't crash the CLI** — feed a synthetic CSV with
   a class-0 event and a non-existent event_id, assert CLI exits
   cleanly with non-zero status and the missing event appears in
   stderr but the class-0 event gets a `metadata.json` with
   `status: "skipped_class_0"`.

### 8.3 Manual verification (the part tests can't cover)

After implementation, a single manual run on a real CISS event from
each of the 4 reconstructable classes (head-on, angle,
single-vehicle, sideswipe). The reviewer looks at the resulting
videos and confirms:

- Head-on: two vehicles close head-on visibly.
- Angle: ego is struck from the side at roughly the recorded PDOF
  angle.
- Single-vehicle: ego strikes a wall.
- Sideswipe: two abreast vehicles make glancing contact.

This is the only way to validate that the parametric variation
actually produces a recognizable crash shape.

### 8.4 What we are NOT testing

- **Pixel-perfect agreement** with the real crash — this is a
  template replay, not a forensic-grade reconstruction.
- **Classifier accuracy** — already covered by the existing ensemble
  CV. Reconstruction consumes predictions; it doesn't evaluate them.
- **BeamNGpy API stability** — out of scope.

---

## 9. Open Questions and Risks

1. **BeamNG.tech v0.38.5.0 specific behavior.** The data generator
   is calibrated to this exact version. If a user has a different
   BeamNG version installed, our calibration will likely still work
   (it auto-detects the heading convention sign) but visual quality
   may differ. We do not gate on version.
2. **OpenCV install footprint.** `opencv-python` is a ~70 MB wheel.
   It's not currently in `.venv`. Acceptable as one new dep for
   Phase 5.
3. **PDOF reliability for sideswipe.** Sideswipe events have low
   peak resultant acceleration (the data generator uses
   `DAMAGE_TRIG['sideswipe'] = 15.0` vs 100-120 for other types
   — see `beamNG.py` lines 73-78). PDOF at low peaks is noisy. We
   accept this: the bounds on sideswipe PDOF offset (±25°) absorb
   the noise.

---

## 10. Reproducibility

```bash
# One-time: add opencv to the venv
.venv\Scripts\python.exe -m pip install opencv-python

# Generate predictions the replayer consumes
.venv\Scripts\python.exe scripts\ensemble.py

# Smoke test
.venv\Scripts\python.exe scripts\reconstruct.py --smoke-test

# Reconstruct one event
.venv\Scripts\python.exe scripts\reconstruct.py --event-id CISS_2019_…

# Reconstruct everything in predictions.csv (skips class-0)
.venv\Scripts\python.exe scripts\reconstruct.py --from-predictions \
        results/ensemble/predictions.csv

# Run unit tests
.venv\Scripts\python.exe -m pytest scripts/tests/test_reconstruct_params.py
```

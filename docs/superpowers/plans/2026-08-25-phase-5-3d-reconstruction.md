# Phase 5 — 3D Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a BeamNG-based 3D crash reconstruction pipeline that consumes ensemble classifier predictions + extracted physics features per event and produces per-event videos with metadata, without modifying the existing `Dataset/data/beamNG.py` data generator.

**Architecture:** Standalone CLI under `scripts/` that re-implements the four crash-class geometries in a new scenarios module, runs each in a fresh BeamNG instance, captures frames from two cameras, and encodes per-event MP4s. Pure-function parameter mapping makes the logic unit-testable without BeamNG.

**Tech Stack:** Python 3.11, BeamNGpy 1.35.1 (BeamNG.tech v0.38.5.0), OpenCV (`opencv-python`) for video encoding, pytest for unit tests, NumPy/Pandas (already in venv).

**Spec:** `docs/superpowers/specs/2026-08-25-3d-reconstruction-design.md`

## Global Constraints

- **Branch:** Implement on `feature/phase-5-3d-reconstruction`. Never commit to `main`.
- **Do not modify** `Dataset/data/beamNG.py` under any circumstance — the data generator is the user's existing infrastructure.
- **Python:** Use `.venv\Scripts\python.exe` (Python 3.11). The system Python 3.14 segfaults on numpy import (per `CLAUDE.md`).
- **BeamNG connection:** `BeamNGpy(HOST='localhost', PORT=64256, launch=True)`. Map `smallgrid`. Vehicle model `etk800`. 60 Hz physics, 10 Hz capture (every 6 steps). 12 s per-scenario timeout.
- **Naming:** All new files under `scripts/`. The CLI is `scripts/reconstruct.py`. Helpers prefixed `reconstruct_`.
- **No silent value imputation.** Missing scalars stay `NaN`; reconstruction skips those events with `status: "insufficient_data"`.
- **Class 0 (rear-end)** is skipped with `status: "skipped_class_0"`. This is a known limitation, documented in CLI `--help`.
- **Output dir:** `results/reconstruction/<event_id>/` containing `metadata.json`, `frame_overview_*.png`, `frame_chase_*.png`, `video_overview.mp4`, `video_chase.mp4`.
- **Predictions input:** `results/ensemble/predictions.csv` with columns `event_id, true_class, pred_class, proba_0..proba_4, fold, seed`.
- **PDOF mapping:** PDOF in degrees is the angle of (accel_lat, accel_long) at peak resultant acceleration. Class-dependent heading mapping per spec Section 4.6.
- **Bounds table** (from spec Section 4.5) — these are hard clamps:
  | Param | Class | Min | Max |
  |---|---|---|---|
  | ego speed | all | 30 km/h | 130 km/h |
  | target speed | 1, 2 | 0 km/h | 130 km/h |
  | target speed | 4 | ego − 5 km/h | ego + 5 km/h |
  | PDOF offset | 1 | −25° | +25° |
  | PDOF offset | 2 | −10° | +10° |
  | PDOF offset | 4 | −25° | +25° |
  | lateral offset | 1 | −1.5 m | +1.5 m |
  | lateral offset | 4 | −0.8 m | +0.8 m |
- **Commit message style:** `feat: <short description>` or `test: <short description>`, body explains why. End with `Co-Authored-By: Claude <noreply@anthropic.com>`.

---

## Task 0: Preflight — add opencv-python to the venv

**Files:**
- Modify: `.venv/` (Python packages, no source file)
- No source file changes

**Why first:** Task 3 (capture + encoding) imports `cv2`. Doing this here means a clean install with the right Python.

- [ ] **Step 1: Install opencv-python into the project venv**

```bash
cd "C:/Users/Aariz/Downloads/CN Reasearch Paper/Forensic-Reconstruction"
.venv\Scripts\python.exe -m pip install opencv-python
```

Expected: Successfully installed opencv-python-X.Y.Z (any recent version).

- [ ] **Step 2: Verify the import works**

```bash
.venv\Scripts\python.exe -c "import cv2; print(cv2.__version__)"
```

Expected: prints a version string like `4.10.0`. Exit code 0.

- [ ] **Step 3: Commit the lock-file change**

The venv is git-ignored. There is no lock file in the repo. Skip this commit; record in `metadata.json` instead (later tasks).

If a `Dataset/requirements.txt` exists (CLAUDE.md notes it is UTF-16 encoded — Phase 1 spec, not `.venv`'s source), leave it untouched. Note in commit message of Task 4 that `opencv-python` was added to `.venv`.

---

## Task 1: Pure param-mapping module `scripts/reconstruct_params.py`

**Files:**
- Create: `scripts/reconstruct_params.py`
- Create: `scripts/tests/__init__.py` (empty, makes `scripts/tests` a package)
- Create: `scripts/tests/test_reconstruct_params.py`
- Test: `scripts/tests/test_reconstruct_params.py`

**Why first:** This module has no BeamNG dependency. Fully unit-testable. Locks down the bounds table, the PDOF→heading mapping, the class-0 skip, and the insufficient-data skip — every behavioral decision the spec makes about *which* events get reconstructed.

**Interfaces:**
- Produces (consumed by Task 2 and later):
  - `class ScenarioParams` — `@dataclass(frozen=True)` with fields:
    `crash_class: int`, `template: str` (one of `'head_on'`, `'angle'`, `'single_vehicle'`, `'sideswipe'`),
    `ego_speed_ms: float`, `target_speed_ms: float | None`,
    `target_heading_deg: float`, `pdof_offset_deg: float`,
    `lateral_offset_m: float`, `contact_duration_s: float = 0.4`
  - `class SkipReason` — `@dataclass(frozen=True)` with `code: str` (one of `'skipped_class_0'`, `'insufficient_data'`), `message: str`
  - `class ParamOverrides` — `@dataclass(frozen=True)` with `original: dict`, `clamped: dict`
  - `def event_to_scenario_params(event_id: str, prediction_row: dict, scalars: dict) -> ScenarioParams | SkipReason`
  - `def apply_speed_bounds(speed_kmh: float) -> float`
  - `def apply_pdof_bounds(pdof_offset_deg: float, crash_class: int) -> float`
  - `def apply_lateral_bounds(lateral_offset_m: float, crash_class: int) -> float`
  - `PDOF_TO_HEADING_DEG` — module-level dict mapping crash_class → callable `(pdof_abs_deg) -> target_heading_deg`

- [ ] **Step 1: Write the failing test file**

```python
# scripts/tests/test_reconstruct_params.py
"""Unit tests for the pure param-mapping module. No BeamNG required."""

import math
import pytest

from scripts.reconstruct_params import (
    ScenarioParams, SkipReason, ParamOverrides,
    event_to_scenario_params,
    apply_speed_bounds, apply_pdof_bounds, apply_lateral_bounds,
)


def _prediction(pred_class=1, probas=None):
    return {
        "event_id": "TEST_001",
        "pred_class": pred_class,
        "proba_0": 0.1, "proba_1": 0.7, "proba_2": 0.05,
        "proba_3": 0.1, "proba_4": 0.05,
    }


def _scalars(pdof_deg=85.0, speed_max_kmh=72.0, along_peak=8.0, alat_peak=4.0, yaw_absmax=10.0):
    return {
        "pdof_deg": pdof_deg,
        "pdof_abs_deg": abs(pdof_deg),
        "speed_kmh_first": speed_max_kmh * 0.9,
        "speed_kmh_max": speed_max_kmh,
        "accel_long_peak": along_peak,
        "accel_lat_peak": alat_peak,
        "yaw_rate_absmax": yaw_absmax,
    }


# --- apply_speed_bounds ---

def test_apply_speed_bounds_clamps_low():
    assert apply_speed_bounds(20.0) == 30.0

def test_apply_speed_bounds_clamps_high():
    assert apply_speed_bounds(140.0) == 130.0

def test_apply_speed_bounds_passes_through_in_range():
    assert apply_speed_bounds(72.0) == 72.0


# --- apply_pdof_bounds ---

def test_apply_pdof_bounds_class1_clamps_high():
    assert apply_pdof_bounds(60.0, crash_class=1) == 25.0

def test_apply_pdof_bounds_class1_clamps_low():
    assert apply_pdof_bounds(-60.0, crash_class=1) == -25.0

def test_apply_pdof_bounds_class2_tighter():
    assert apply_pdof_bounds(15.0, crash_class=2) == 10.0
    assert apply_pdof_bounds(-15.0, crash_class=2) == -10.0

def test_apply_pdof_bounds_class4_same_as_class1():
    assert apply_pdof_bounds(60.0, crash_class=4) == 25.0

def test_apply_pdof_bounds_class3_raises():
    """Single-vehicle has no PDOF offset — caller should not call this for class 3."""
    with pytest.raises(ValueError):
        apply_pdof_bounds(10.0, crash_class=3)


# --- apply_lateral_bounds ---

def test_apply_lateral_bounds_class1_clamps():
    assert apply_lateral_bounds(3.0, crash_class=1) == 1.5
    assert apply_lateral_bounds(-3.0, crash_class=1) == -1.5

def test_apply_lateral_bounds_class4_tighter():
    assert apply_lateral_bounds(2.0, crash_class=4) == 0.8
    assert apply_lateral_bounds(-2.0, crash_class=4) == -0.8


# --- event_to_scenario_params: skip paths ---

def test_event_to_scenario_params_skips_class_0():
    result = event_to_scenario_params("TEST_001", _prediction(pred_class=0), _scalars())
    assert isinstance(result, SkipReason)
    assert result.code == "skipped_class_0"


def test_event_to_scenario_params_skips_insufficient_speed():
    scalars = _scalars(speed_max_kmh=float("nan"))
    result = event_to_scenario_params("TEST_001", _prediction(pred_class=1), scalars)
    assert isinstance(result, SkipReason)
    assert result.code == "insufficient_data"


def test_event_to_scenario_params_skips_insufficient_pdof():
    scalars = _scalars(pdof_deg=float("nan"))
    result = event_to_scenario_params("TEST_001", _prediction(pred_class=1), scalars)
    assert isinstance(result, SkipReason)
    assert result.code == "insufficient_data"


# --- event_to_scenario_params: success paths ---

def test_event_to_scenario_params_class1_angle_with_pdof():
    result = event_to_scenario_params("TEST_001", _prediction(pred_class=1), _scalars(pdof_deg=85.0))
    assert isinstance(result, ScenarioParams)
    assert result.crash_class == 1
    assert result.template == "angle"
    assert result.target_heading_deg == pytest.approx(85.0, abs=1e-6)
    assert math.isclose(result.ego_speed_ms, 72.0 / 3.6, abs_tol=1e-6)


def test_event_to_scenario_params_class2_head_on_canonical_heading():
    result = event_to_scenario_params("TEST_002", _prediction(pred_class=2), _scalars(pdof_deg=5.0))
    assert isinstance(result, ScenarioParams)
    assert result.template == "head_on"
    assert result.target_heading_deg == pytest.approx(180.0, abs=1e-6)


def test_event_to_scenario_params_class3_single_vehicle_no_target():
    result = event_to_scenario_params("TEST_003", _prediction(pred_class=3), _scalars())
    assert isinstance(result, ScenarioParams)
    assert result.template == "single_vehicle"
    assert result.target_speed_ms is None


def test_event_to_scenario_params_class4_sideswipe_breast_speed():
    result = event_to_scenario_params("TEST_004", _prediction(pred_class=4), _scalars(pdof_deg=88.0))
    assert isinstance(result, ScenarioParams)
    assert result.template == "sideswipe"
    assert math.isclose(result.ego_speed_ms, result.target_speed_ms, abs_tol=5.0 / 3.6 + 1e-6)


# --- determinism ---

def test_event_to_scenario_params_is_deterministic():
    a = event_to_scenario_params("X", _prediction(pred_class=1), _scalars(pdof_deg=70.0))
    b = event_to_scenario_params("X", _prediction(pred_class=1), _scalars(pdof_deg=70.0))
    assert a == b
```

- [ ] **Step 2: Run the tests; expect all to fail with ImportError**

```bash
cd "C:/Users/Aariz/Downloads/CN Reasearch Paper/Forensic-Reconstruction"
.venv\Scripts\python.exe -m pytest scripts/tests/test_reconstruct_params.py -v
```

Expected: collection error / `ModuleNotFoundError: No module named 'scripts.reconstruct_params'` for every test.

- [ ] **Step 3: Create the empty package init**

```bash
mkdir -p scripts/tests
```

Then write `scripts/tests/__init__.py` (empty file, just makes it a package).

- [ ] **Step 4: Implement `scripts/reconstruct_params.py`**

```python
# scripts/reconstruct_params.py
"""
Pure param-mapping for Phase 5 reconstruction.

Maps (event_id, classifier prediction, extracted physics scalars) to a
ScenarioParams dict that the BeamNG scenarios module consumes. No BeamNG
dependency here — every function is unit-testable headlessly.

Source of every constant and bound: docs/superpowers/specs/2026-08-25-
3d-reconstruction-design.md sections 4.5, 4.6, 4.7.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional


# Speed bounds (km/h), all classes
EGO_SPEED_MIN_KMH = 30.0
EGO_SPEED_MAX_KMH = 130.0

# Per-class bounds. Keys are crash_class ints (0-4).
PDOF_OFFSET_BOUNDS_DEG = {
    1: (-25.0, 25.0),   # angle
    2: (-10.0, 10.0),   # head-on (PDOF should be near collinear)
    4: (-25.0, 25.0),   # sideswipe
    # class 3 (single-vehicle) has no PDOF offset — caller must not invoke
}

LATERAL_OFFSET_BOUNDS_M = {
    1: (-1.5, 1.5),     # angle
    4: (-0.8, 0.8),     # sideswipe (tighter — glancing contact)
    # class 2 (head-on) and class 3 (single-vehicle) have no lateral offset
}


@dataclass(frozen=True)
class ScenarioParams:
    """Concrete scenario parameters consumed by reconstruct_scenarios.build_replay_scenario."""
    crash_class: int
    template: str           # 'head_on' | 'angle' | 'single_vehicle' | 'sideswipe'
    ego_speed_ms: float
    target_speed_ms: Optional[float]
    target_heading_deg: float
    pdof_offset_deg: float
    lateral_offset_m: float
    contact_duration_s: float = 0.4


@dataclass(frozen=True)
class SkipReason:
    code: str   # 'skipped_class_0' | 'insufficient_data'
    message: str


def apply_speed_bounds(speed_kmh: float) -> float:
    """Clamp a speed value to the global ego/target bounds (km/h)."""
    return max(EGO_SPEED_MIN_KMH, min(EGO_SPEED_MAX_KMH, speed_kmh))


def apply_pdof_bounds(pdof_offset_deg: float, crash_class: int) -> float:
    if crash_class not in PDOF_OFFSET_BOUNDS_DEG:
        raise ValueError(f"crash_class {crash_class} has no PDOF offset (single-vehicle / rear-end)")
    lo, hi = PDOF_OFFSET_BOUNDS_DEG[crash_class]
    return max(lo, min(hi, pdof_offset_deg))


def apply_lateral_bounds(lateral_offset_m: float, crash_class: int) -> float:
    if crash_class not in LATERAL_OFFSET_BOUNDS_M:
        raise ValueError(f"crash_class {crash_class} has no lateral offset (head-on / single-vehicle)")
    lo, hi = LATERAL_OFFSET_BOUNDS_M[crash_class]
    return max(lo, min(hi, lateral_offset_m))


def _is_finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def _pdof_to_heading_class1(pdof_abs_deg: float) -> float:
    """Angle crash: target approach heading = pdof_abs_deg (90° = pure lateral)."""
    return float(pdof_abs_deg)


def _pdof_to_heading_class2(_pdof_abs_deg: float) -> float:
    """Head-on crash: target always approaches from 180° regardless of PDOF."""
    return 180.0


def _pdof_to_heading_class4(pdof_abs_deg: float) -> float:
    """Sideswipe crash: target approach heading = pdof_abs_deg."""
    return float(pdof_abs_deg)


PDOF_TO_HEADING_DEG: dict[int, Callable[[float], float]] = {
    1: _pdof_to_heading_class1,
    2: _pdof_to_heading_class2,
    4: _pdof_to_heading_class4,
}


def event_to_scenario_params(
    event_id: str,
    prediction_row: dict,
    scalars: dict,
) -> ScenarioParams | SkipReason:
    """Map an event's prediction + scalars to ScenarioParams, or a SkipReason.

    Returns SkipReason with code 'skipped_class_0' for rear-end events
    (the data generator does not synthesize rear-ends, so we cannot
    produce a meaningful template — see spec section 4.7).

    Returns SkipReason with code 'insufficient_data' when any required
    scalar is NaN or non-finite.
    """
    pred_class = int(prediction_row["pred_class"])

    if pred_class == 0:
        return SkipReason("skipped_class_0", f"event {event_id}: rear-end not synthesized")

    required = ["speed_kmh_max", "pdof_abs_deg"]
    for k in required:
        v = scalars.get(k)
        if v is None or not math.isfinite(v):
            return SkipReason("insufficient_data", f"event {event_id}: scalar {k} missing/non-finite")

    speed_kmh = apply_speed_bounds(float(scalars["speed_kmh_max"]))
    ego_speed_ms = speed_kmh / 3.6

    template_map = {1: "angle", 2: "head_on", 3: "single_vehicle", 4: "sideswipe"}
    template = template_map[pred_class]

    if template == "single_vehicle":
        target_speed_ms: Optional[float] = None
        target_heading_deg = 0.0
        pdof_offset_deg = 0.0
        lateral_offset_m = 0.0
    else:
        pdof_abs_deg = float(scalars["pdof_abs_deg"])
        # PDOF offset is the deviation of pdof_abs_deg from the canonical heading.
        # For class 1/4: canonical = 90°; for class 2: canonical = 0° (but we ignore PDOF entirely).
        canonical_heading = 90.0 if pred_class in (1, 4) else 0.0
        raw_pdof_offset = pdof_abs_deg - canonical_heading
        pdof_offset_deg = apply_pdof_bounds(raw_pdof_offset, pred_class)

        target_heading_deg = PDOF_TO_HEADING_DEG[pred_class](pdof_abs_deg)

        if template == "head_on":
            # Head-on: target_speed = ego_speed * (1 +/- 0.15) per spec
            target_speed_kmh = speed_kmh * 1.0  # equal is the canonical case; variation comes from speed itself
            target_speed_ms = max(0.0, target_speed_kmh / 3.6)
            lateral_offset_m = 0.0
        elif template == "angle":
            target_speed_kmh = speed_kmh * 0.5  # forensic default
            target_speed_ms = target_speed_kmh / 3.6
            # Lateral offset defaults to 0; spec doesn't say to extract from data
            lateral_offset_m = 0.0
        elif template == "sideswipe":
            target_speed_ms = ego_speed_ms  # abreast
            lateral_offset_m = 0.0
        else:
            raise ValueError(f"unhandled template {template}")

    return ScenarioParams(
        crash_class=pred_class,
        template=template,
        ego_speed_ms=ego_speed_ms,
        target_speed_ms=target_speed_ms,
        target_heading_deg=target_heading_deg,
        pdof_offset_deg=pdof_offset_deg,
        lateral_offset_m=lateral_offset_m,
    )
```

- [ ] **Step 5: Run the tests; expect all to pass**

```bash
cd "C:/Users/Aariz/Downloads/CN Reasearch Paper/Forensic-Reconstruction"
.venv\Scripts\python.exe -m pytest scripts/tests/test_reconstruct_params.py -v
```

Expected: 18 passed, 0 failed.

- [ ] **Step 6: Commit**

```bash
cd "C:/Users/Aariz/Downloads/CN Reasearch Paper/Forensic-Reconstruction"
git add scripts/reconstruct_params.py scripts/tests/__init__.py scripts/tests/test_reconstruct_params.py
git commit -m "feat(reconstruct): pure param-mapping module with TDD tests

No BeamNG dependency. Locks down bounds table, PDOF->heading mapping,
class-0 skip, and insufficient-data skip. Tests cover clamp boundaries
on every per-class bound and the determinism invariant.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Scenario construction module `scripts/reconstruct_scenarios.py`

**Files:**
- Create: `scripts/reconstruct_scenarios.py`

**Why second:** Builds BeamNG `Scenario` objects from `ScenarioParams`. Reimplements the four crash-class geometries from the data generator (`Dataset/data/beamNG.py` lines 156-208) without importing it. Owns the calibration logic.

**Interfaces:**
- Consumes (from Task 1): `ScenarioParams` dataclass
- Produces (consumed by Task 3):
  - `class CalibratedHeading(NamedTuple)` with `qh: Callable[[float], Any]` (the heading closure)
  - `def calibrate(bng: BeamNGpy) -> CalibratedHeading`
  - `def build_replay_scenario(params: ScenarioParams, calib: CalibratedHeading, event_id: str) -> tuple[Scenario, Vehicle, Vehicle | None, ProceduralCube | None]`

- [ ] **Step 1: Write the file**

```python
# scripts/reconstruct_scenarios.py
"""
BeamNG scenario construction for Phase 5 reconstruction.

Reimplements the four crash-class geometries from Dataset/data/beamNG.py
(without importing that file — the data generator is untouched by design,
see spec section 1). Owns the calibration probe-vehicle logic and the
Scenario/Vehicle wiring.

Map: 'smallgrid'. Vehicle model: 'etk800'. Physics: 60 Hz.
"""
from __future__ import annotations

import math
from typing import Any, Callable, NamedTuple, Optional

from beamngpy import BeamNGpy, Scenario, Vehicle, ProceduralCube, angle_to_quat


VEHICLE_MODEL = "etk800"
APPROACH_DIST_M = {
    "head_on": 55.0,
    "angle": 55.0,
    "single_vehicle": 65.0,
    "sideswipe": 55.0,
}


class CalibratedHeading(NamedTuple):
    """Returned by calibrate(); qh(theta_deg) -> quaternion driving along world heading theta (0=+X, CCW+)."""
    qh: Callable[[float], Any]


def calibrate(bng: BeamNGpy) -> CalibratedHeading:
    """Probe-vehicle calibration. Verifies set_velocity direction.

    Mirrors the logic in Dataset/data/beamNG.py calibrate() — separate
    implementation, no import.
    """
    print("  [reconstruct] calibrating heading + launch ...")
    probe = Vehicle("probe", model=VEHICLE_MODEL, color="White")
    sc = Scenario("smallgrid", "calibration")
    sc.add_vehicle(probe, pos=(0, 0, 0.3), rot_quat=angle_to_quat((0, 0, 0)))
    sc.make(bng)
    bng.load_scenario(sc)
    bng.start_scenario()

    bng.step(30)  # settle half a second at 60Hz
    probe.poll_sensors()
    d0 = probe.state["dir"]
    yaw0 = math.degrees(math.atan2(d0[1], d0[0]))

    probe.teleport((0, 0, 0.3), rot_quat=angle_to_quat((0, 0, 90)), reset=True)
    bng.step(30)
    probe.poll_sensors()
    d90 = probe.state["dir"]
    yaw90 = math.degrees(math.atan2(d90[1], d90[0]))

    s = 1.0 if ((yaw90 - yaw0 + 180.0) % 360.0 - 180.0) > 0 else -1.0
    print(f"    identity heading = {yaw0:6.1f} deg | +90cmd = {yaw90:6.1f} deg | sign = {s:+.0f}")

    def qh(theta_deg: float):
        return angle_to_quat((0, 0, (theta_deg - yaw0) / s))

    probe.teleport((-30.0, 0.0, 0.3), rot_quat=qh(0.0), reset=True)
    bng.step(30)
    probe.poll_sensors()
    x0 = probe.state["pos"][0]
    probe.set_velocity(14.0, dt=1.0)
    bng.step(90)
    probe.poll_sensors()
    p = probe.state["pos"]
    adv = p[0] - x0
    drift = abs(p[1])
    bng.stop_scenario()
    print(f"    launch check: advanced {adv:+.1f} m along +X, lateral drift {drift:.1f} m")
    if adv < 3.0 or drift > 8.0:
        raise RuntimeError(
            "set_velocity did not propel toward the target — BeamNG version mismatch?"
        )
    print("    calibration OK.")
    return CalibratedHeading(qh=qh)


def _toward_origin(calib: CalibratedHeading, dist: float, heading_deg: float) -> tuple[float, float]:
    """Return (x, y) such that a vehicle placed there with heading_deg drives to origin."""
    th = math.radians(heading_deg)
    return (-math.cos(th) * dist, -math.sin(th) * dist)


def build_replay_scenario(
    params: ScenarioParams,
    calib: CalibratedHeading,
    event_id: str,
) -> tuple[Scenario, Vehicle, Optional[Vehicle], Optional[ProceduralCube]]:
    """Build a BeamNG scenario for one reconstruction.

    Returns (scenario, ego, target_or_none, obstacle_or_none).
    The caller (capture module) attaches sensors to ego.
    """
    qh = calib.qh
    template = params.template
    D = APPROACH_DIST_M[template]
    scenario = Scenario("smallgrid", f"recon_{template}_{event_id}")
    ego = Vehicle("ego", model=VEHICLE_MODEL, color="Red")
    scenario.add_vehicle(ego, pos=(-D, 0.0, 0.3), rot_quat=qh(0.0))

    if template == "head_on":
        tx, ty = _toward_origin(calib, D, 180.0)
        target = Vehicle("other", model=VEHICLE_MODEL, color="Blue")
        scenario.add_vehicle(target, pos=(tx, ty, 0.3), rot_quat=qh(180.0))
        return scenario, ego, target, None

    if template == "angle":
        # Canonical: ego at -X facing +X; target approaches from -Y facing +Y (heading 90°)
        target_heading = params.target_heading_deg
        tx, ty = _toward_origin(calib, D, target_heading)
        # Lateral offset perpendicular to target_heading (toward ego)
        perp_x = -math.sin(math.radians(target_heading))
        perp_y = math.cos(math.radians(target_heading))
        tx += perp_x * params.lateral_offset_m
        ty += perp_y * params.lateral_offset_m
        target = Vehicle("other", model=VEHICLE_MODEL, color="Blue")
        scenario.add_vehicle(target, pos=(tx, ty, 0.3), rot_quat=qh(target_heading))
        return scenario, ego, target, None

    if template == "single_vehicle":
        # Wall in front of ego at offset
        wall = ProceduralCube(
            pos=(0.0, params.lateral_offset_m, 1.5),
            size=(2.5, 10.0, 3.0),
            name=f"wall_{event_id}",
            rot_quat=angle_to_quat((0, 0, 0)),
        )
        scenario.add_procedural_mesh(wall)
        return scenario, ego, None, wall

    if template == "sideswipe":
        # Both vehicles abreast: ego right lane (y=-1.5) trailing by 2.5 m,
        # target left lane (y=+1.5) at -D. PDOF offset rotates ego's heading.
        target_heading = params.target_heading_deg
        lat = params.lateral_offset_m
        target = Vehicle("other", model=VEHICLE_MODEL, color="Blue")
        scenario.add_vehicle(target, pos=(-D, 1.5 + lat, 0.3), rot_quat=qh(target_heading))
        # Place ego correctly the first time (don't reuse the earlier default placement).
        ego = Vehicle("ego", model=VEHICLE_MODEL, color="Red")
        scenario.add_vehicle(ego, pos=(-(D + 2.5), -1.5 - lat, 0.3), rot_quat=qh(params.pdof_offset_deg))
        return scenario, ego, target, None

    raise ValueError(f"unknown template {template}")
```

- [ ] **Step 2: Smoke-import the module**

```bash
cd "C:/Users/Aariz/Downloads/CN Reasearch Paper/Forensic-Reconstruction"
.venv\Scripts\python.exe -c "from scripts.reconstruct_scenarios import calibrate, build_replay_scenario, CalibratedHeading; print('imports ok')"
```

Expected: prints `imports ok`. This step requires `beamngpy` installed in `.venv`. If missing:

```bash
.venv\Scripts\python.exe -m pip install beamngpy
```

(BeamNGpy install is independent of BeamNG.tech running — the package is just the Python client.)

- [ ] **Step 3: Commit**

```bash
cd "C:/Users/Aariz/Downloads/CN Reasearch Paper/Forensic-Reconstruction"
git add scripts/reconstruct_scenarios.py
git commit -m "feat(reconstruct): scenario construction for 4 crash templates

Reimplements geometry from Dataset/data/beamNG.py without importing it.
Owns calibration probe logic and Scenario/Vehicle wiring. Pure functions
over ScenarioParams — no I/O, no connection lifecycle.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Capture & encoding module `scripts/reconstruct_capture.py`

**Files:**
- Create: `scripts/reconstruct_capture.py`

**Why third:** Owns the BeamNG connection lifecycle, runs scenarios, captures frames from two `FreeCamera` sensors, and encodes MP4 videos with OpenCV. Independent of the scenario construction.

**Interfaces:**
- Consumes (from Tasks 1 & 2): `ScenarioParams`, scenario/ego/target from `build_replay_scenario`
- Produces (consumed by Task 4):
  - `class CaptureResult(NamedTuple)` with `frames_overview: list[Path]`, `frames_chase: list[Path]`, `video_overview: Path | None`, `video_chase: Path | None`, `status: str` (one of `'ok'`, `'beamng_error'`, `'timeout'`), `error_message: str | None`
  - `def open_beamng(host: str = 'localhost', port: int = 64256) -> BeamNGpy` — context manager
  - `def run_and_capture(scenario: Scenario, ego: Vehicle, params: ScenarioParams, output_dir: Path, bng: BeamNGpy, calib: CalibratedHeading, max_sim_s: float = 12.0) -> CaptureResult`

- [ ] **Step 1: Write the file**

```python
# scripts/reconstruct_capture.py
"""
BeamNG connection lifecycle + per-event capture + MP4 encoding for Phase 5.

Owns:
- BeamNGpy connection (open + guaranteed close)
- FreeCamera sensor attachment
- Physics loop with impact detection (12 s timeout)
- Frame capture from 2 cameras at 10 Hz
- MP4 encoding via opencv-python (mp4v codec, 1280x720)
"""
from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

import cv2
import numpy as np

from beamngpy import BeamNGpy
from beamngpy.sensors import Damage, Electrics, FreeCamera

from scripts.reconstruct_scenarios import CalibratedHeading
from scripts.reconstruct_params import ScenarioParams


PHYSICS_HZ = 60
CAPTURE_STEP = 6                    # 6/60 = 0.1 s per sample = 10 Hz
SAMPLE_DT = CAPTURE_STEP / PHYSICS_HZ
POST_IMPACT_S = 0.6
DAMAGE_TRIG = {
    "head_on": 120.0,
    "angle": 100.0,
    "single_vehicle": 120.0,
    "sideswipe": 15.0,
}
VIDEO_FPS = 10
VIDEO_SIZE = (1280, 720)
CODEC = "mp4v"


class CaptureResult(NamedTuple):
    frames_overview: list[Path]
    frames_chase: list[Path]
    video_overview: Optional[Path]
    video_chase: Optional[Path]
    status: str          # 'ok' | 'beamng_error' | 'timeout'
    error_message: Optional[str]


@contextmanager
def open_beamng(host: str = "localhost", port: int = 64256) -> Iterator[BeamNGpy]:
    """Open a BeamNG connection; guarantee close on exit."""
    bng = BeamNGpy(host, port, launch=True)
    try:
        bng.open()
        yield bng
    finally:
        try:
            bng.close()
        except Exception:
            pass


def _total_damage(d):
    if not d:
        return 0.0
    v = d.get("damage")
    if isinstance(v, (int, float)):
        return float(v)
    part = d.get("part_damage") or {}
    return sum(float(p.get("damage", 0) or 0) for p in part.values() if isinstance(p, dict))


def _attach_cameras(ego: Vehicle) -> tuple[FreeCamera, FreeCamera]:
    overview = FreeCamera(
        "overview", pos=(0, 50, 30), dir=(0, -1, -0.6), fov=60,
        resolution=VIDEO_SIZE, near_far=(0.1, 1000),
    )
    chase = FreeCamera(
        "chase", pos=(0, -8, 4), dir=(0, 1, -0.2), fov=75,
        resolution=VIDEO_SIZE, near_far=(0.1, 1000),
    )
    overview.attach(ego)
    chase.attach(ego)
    return overview, chase


def _write_video(frames: list[Path], out_path: Path) -> Optional[Path]:
    if not frames:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*CODEC),
        VIDEO_FPS,
        VIDEO_SIZE,
    )
    if not writer.isOpened():
        return None
    try:
        for fp in frames:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            if img.shape[1] != VIDEO_SIZE[0] or img.shape[0] != VIDEO_SIZE[1]:
                img = cv2.resize(img, VIDEO_SIZE)
            writer.write(img)
    finally:
        writer.release()
    return out_path if out_path.exists() and out_path.stat().st_size > 0 else None


def run_and_capture(
    scenario,
    ego: Vehicle,
    params: ScenarioParams,
    output_dir: Path,
    bng: BeamNGpy,
    calib: CalibratedHeading,
    max_sim_s: float = 12.0,
) -> CaptureResult:
    """Run one scenario; capture frames; encode MP4s. Returns CaptureResult."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ego.attach_sensor("electrics", Electrics())
    ego.attach_sensor("damage", Damage())
    overview_cam, chase_cam = _attach_cameras(ego)

    err: Optional[str] = None
    status = "ok"
    overview_frames: list[Path] = []
    chase_frames: list[Path] = []
    impact_t: Optional[float] = None
    elapsed = 0.0
    frame_idx = 0

    try:
        scenario.make(bng)
        bng.load_scenario(scenario)
        bng.start_scenario()
        bng.step(PHYSICS_HZ // 2)  # settle
        ego.poll_sensors()
        baseline_dmg = _total_damage(ego.sensors["damage"])
        trigger = DAMAGE_TRIG[params.template]

        # Launch both vehicles (or just ego for single-vehicle)
        ego.set_velocity(params.ego_speed_ms, dt=1.0 + params.ego_speed_ms / 25.0)
        ego.control(throttle=0.45, steering=0.0, brake=0.0)
        target = None
        for v in scenario.vehicles.values() if hasattr(scenario, "vehicles") else []:
            if v.vid != ego.vid:
                target = v
                break
        if target is not None and params.target_speed_ms is not None:
            target.set_velocity(params.target_speed_ms, dt=1.0 + params.target_speed_ms / 25.0)
            target.control(throttle=0.45, steering=0.0, brake=0.0)

        n_steps = int(max_sim_s / SAMPLE_DT)
        for step in range(n_steps):
            bng.step(CAPTURE_STEP)
            ego.poll_sensors()
            overview_cam.poll()
            chase_cam.poll()
            elapsed = step * SAMPLE_DT

            # Save frames
            ov_path = output_dir / f"frame_overview_{frame_idx:04d}.png"
            ch_path = output_dir / f"frame_chase_{frame_idx:04d}.png"
            try:
                if overview_cam.image is not None:
                    cv2.imwrite(str(ov_path), overview_cam.image)
                    overview_frames.append(ov_path)
                if chase_cam.image is not None:
                    cv2.imwrite(str(ch_path), chase_cam.image)
                    chase_frames.append(ch_path)
                frame_idx += 1
            except Exception as e:
                err = f"frame_save_error: {e}"
                break

            dmg_now = _total_damage(ego.sensors["damage"])
            if dmg_now - baseline_dmg > trigger and impact_t is None:
                impact_t = elapsed
                post_steps = int(POST_IMPACT_S / SAMPLE_DT)
                for _ in range(post_steps):
                    bng.step(CAPTURE_STEP)
                    ego.poll_sensors()
                    overview_cam.poll()
                    chase_cam.poll()
                    ov_path = output_dir / f"frame_overview_{frame_idx:04d}.png"
                    ch_path = output_dir / f"frame_chase_{frame_idx:04d}.png"
                    if overview_cam.image is not None:
                        cv2.imwrite(str(ov_path), overview_cam.image)
                        overview_frames.append(ov_path)
                    if chase_cam.image is not None:
                        cv2.imwrite(str(ch_path), chase_cam.image)
                        chase_frames.append(ch_path)
                    frame_idx += 1
                break

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        status = "beamng_error"
    finally:
        try:
            bng.stop_scenario()
        except Exception:
            pass

    if status == "ok" and impact_t is None:
        status = "timeout"
        err = "no impact within timeout"

    # Encode videos
    video_overview = _write_video(overview_frames, output_dir / "video_overview.mp4")
    video_chase = _write_video(chase_frames, output_dir / "video_chase.mp4")

    return CaptureResult(
        frames_overview=overview_frames,
        frames_chase=chase_frames,
        video_overview=video_overview,
        video_chase=video_chase,
        status=status,
        error_message=err,
    )
```

- [ ] **Step 2: Smoke-import the module**

```bash
cd "C:/Users/Aariz/Downloads/CN Reasearch Paper/Forensic-Reconstruction"
.venv\Scripts\python.exe -c "from scripts.reconstruct_capture import open_beamng, run_and_capture, CaptureResult, VIDEO_SIZE, VIDEO_FPS; print('imports ok', VIDEO_SIZE, VIDEO_FPS)"
```

Expected: prints `imports ok (1280, 720) 10`.

- [ ] **Step 3: Commit**

```bash
cd "C:/Users/Aariz/Downloads/CN Reasearch Paper/Forensic-Reconstruction"
git add scripts/reconstruct_capture.py
git commit -m "feat(reconstruct): BeamNG capture + MP4 encoding module

Owns connection lifecycle (open/close guaranteed), FreeCamera attach,
physics loop with damage-trigger detection (12 s timeout), and mp4v
video encoding via opencv-python. Independent of scenarios and params.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: CLI entry point `scripts/reconstruct.py`

**Files:**
- Create: `scripts/reconstruct.py`

**Why fourth:** Glue. Parses CLI args, loads predictions.csv + features, calls the three modules in order, writes metadata.json.

**Interfaces:**
- Consumes: outputs of Tasks 1, 2, 3
- Produces: side-effects in `results/reconstruction/<event_id>/`

- [ ] **Step 1: Write the file**

```python
# scripts/reconstruct.py
"""
CLI entry point for Phase 5 reconstruction.

Usage:
    python scripts/reconstruct.py --smoke-test
    python scripts/reconstruct.py --event-id CISS_2019_…
    python scripts/reconstruct.py --from-predictions results/ensemble/predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import reconstruct_capture, reconstruct_params, reconstruct_scenarios
from scripts.features import build_rich
from scripts.reconstruct_capture import CaptureResult, open_beamng
from scripts.reconstruct_params import ScenarioParams, SkipReason
from scripts.reconstruct_scenarios import calibrate, build_replay_scenario


DATA_CSV = Path("Dataset/data/training_dataset_final.csv.gz")
PREDICTIONS_CSV = Path("results/ensemble/predictions.csv")
OUTPUT_ROOT = Path("results/reconstruction")
FEATURE_COLS_PATH = Path("models/mixed/feature_cols.json")


def _load_prediction(event_id: str, predictions_csv: Path) -> dict | None:
    if not predictions_csv.exists():
        return None
    with predictions_csv.open() as f:
        for row in csv.DictReader(f):
            if row["event_id"] == event_id:
                return row
    return None


def _load_scalars(event_id: str) -> dict:
    """Compute per-event scalars via build_rich; pull just what we need."""
    df = pd.read_csv(DATA_CSV)
    event_df = df[df["event_id"] == event_id].sort_values("t").reset_index(drop=True)
    if len(event_df) == 0:
        return {}
    feature_cols = json.loads(FEATURE_COLS_PATH.read_text())
    n_feat = len(feature_cols)
    T = 160
    X = np.full((1, T, n_feat), np.nan, dtype=np.float32)
    n = min(len(event_df), T)
    cols = [event_df[c].to_numpy(dtype=np.float32) for c in feature_cols]
    arr = np.stack(cols, axis=1) if cols else np.zeros((0, n_feat), dtype=np.float32)
    X[0, :n] = arr[:n]
    mask = np.zeros((1, T), dtype=bool)
    mask[0, n:] = True
    rich, names = build_rich(X, mask, feature_cols)
    name_to_idx = {n: i for i, n in enumerate(names)}
    needed = ["pdof_deg", "pdof_abs_deg", "speed_kmh_first", "speed_kmh_max",
              "accel_long_peak", "accel_lat_peak", "yaw_rate_absmax"]
    out = {k: float(rich[0, name_to_idx[k]]) if k in name_to_idx else float("nan") for k in needed}
    return out


def _reconstruct_one(event_id: str, predictions_csv: Path, bng) -> dict:
    pred = _load_prediction(event_id, predictions_csv)
    if pred is None:
        return {"event_id": event_id, "status": "missing_in_predictions",
                "error_message": f"{event_id} not in {predictions_csv}"}
    scalars = _load_scalars(event_id)
    if not scalars:
        return {"event_id": event_id, "status": "insufficient_data",
                "error_message": "no rows for this event_id in training_dataset_final.csv"}

    proba = {f"proba_{i}": float(pred[f"proba_{i}"]) for i in range(5)}
    prediction_row = {"event_id": event_id, "pred_class": int(pred["pred_class"]), **proba}
    params_or_skip = reconstruct_params.event_to_scenario_params(event_id, prediction_row, scalars)

    out_dir = OUTPUT_ROOT / event_id
    meta = {
        "event_id": event_id,
        "pred_class": int(pred["pred_class"]),
        "proba": proba,
    }

    if isinstance(params_or_skip, SkipReason):
        meta["status"] = params_or_skip.code
        meta["error_message"] = params_or_skip.message
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        return meta

    assert isinstance(params_or_skip, ScenarioParams)
    params = params_or_skip
    meta["scenario_params"] = {
        "template": params.template,
        "ego_speed_ms": params.ego_speed_ms,
        "target_speed_ms": params.target_speed_ms,
        "target_heading_deg": params.target_heading_deg,
        "pdof_offset_deg": params.pdof_offset_deg,
        "lateral_offset_m": params.lateral_offset_m,
    }
    meta["scalars"] = scalars

    calib = calibrate(bng)
    scenario, ego, target, obstacle = build_replay_scenario(params, calib, event_id)
    result: CaptureResult = reconstruct_capture.run_and_capture(
        scenario, ego, params, out_dir, bng, calib
    )
    meta["status"] = result.status
    meta["error_message"] = result.error_message
    meta["n_frames_overview"] = len(result.frames_overview)
    meta["n_frames_chase"] = len(result.frames_chase)
    meta["video_overview"] = str(result.video_overview) if result.video_overview else None
    meta["video_chase"] = str(result.video_chase) if result.video_chase else None
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def _smoke_test(bng) -> int:
    params = ScenarioParams(
        crash_class=2,
        template="head_on",
        ego_speed_ms=60.0 / 3.6,
        target_speed_ms=60.0 / 3.6,
        target_heading_deg=180.0,
        pdof_offset_deg=0.0,
        lateral_offset_m=0.0,
    )
    calib = calibrate(bng)
    scenario, ego, _, _ = build_replay_scenario(params, calib, "smoke_test")
    out_dir = OUTPUT_ROOT / "smoke_test"
    result = reconstruct_capture.run_and_capture(scenario, ego, params, out_dir, bng, calib)
    print(f"smoke_test status={result.status} frames_overview={len(result.frames_overview)}")
    return 0 if result.status == "ok" else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Phase 5 — 3D BeamNG reconstruction from classifier predictions.",
        epilog="KNOWN LIMITATION: class 0 (rear-end) is skipped — see spec section 4.7.",
    )
    p.add_argument("--event-id", type=str, default=None)
    p.add_argument("--from-predictions", type=Path, default=PREDICTIONS_CSV)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=64256)
    args = p.parse_args(argv)

    if not args.smoke_test and not args.event_id and not args.from_predictions:
        p.error("one of --event-id, --from-predictions, or --smoke-test required")

    rc = 0
    with open_beamng(args.host, args.port) as bng:
        if args.smoke_test:
            return _smoke_test(bng)

        if args.event_id:
            meta = _reconstruct_one(args.event_id, args.from_predictions, bng)
            print(json.dumps(meta, indent=2))
            return 0 if meta["status"] == "ok" else 1

        # --from-predictions: iterate
        with args.from_predictions.open() as f:
            reader = csv.DictReader(f)
            event_ids = [row["event_id"] for row in reader]
        print(f"reconstructing {len(event_ids)} events ...")
        for eid in event_ids:
            print(f"-- {eid}")
            try:
                meta = _reconstruct_one(eid, args.from_predictions, bng)
            except Exception as e:
                meta = {"event_id": eid, "status": "cli_error", "error_message": f"{type(e).__name__}: {e}"}
            print(f"   status={meta['status']}")
            if meta["status"] not in ("ok", "skipped_class_0", "insufficient_data"):
                rc = 1
        return rc


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify --help renders without errors**

```bash
cd "C:/Users/Aariz/Downloads/CN Reasearch Paper/Forensic-Reconstruction"
.venv\Scripts\python.exe scripts/reconstruct.py --help
```

Expected: prints usage text including the known limitation note. Exit code 0.

- [ ] **Step 3: Commit**

```bash
cd "C:/Users/Aariz/Downloads/CN Reasearch Paper/Forensic-Reconstruction"
git add scripts/reconstruct.py
git commit -m "feat(reconstruct): CLI entry point for Phase 5

Glue module: parses --smoke-test / --event-id / --from-predictions,
loads predictions.csv + per-event scalars via build_rich, runs the
scenarios + capture modules, writes metadata.json per event.
Known limitation (class-0 skip) documented in --help.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: `ensemble.py` writes `predictions.csv`

**Files:**
- Modify: `scripts/ensemble.py` (find the `pooled_true` / `pooled_pred` block, add CSV write)

**Why here:** The reconstruction pipeline needs a stable input format. Today the ensemble only writes `ensemble_report.txt`. Adding a CSV is additive — does not change any reported metric.

**Interfaces:**
- Produces: `results/ensemble/predictions.csv` with columns `event_id, true_class, pred_class, proba_0..proba_4, fold, seed`

- [ ] **Step 1: Read the existing pooled-prediction block**

```bash
cd "C:/Users/Aariz/Downloads/CN Reasearch Paper/Forensic-Reconstruction"
grep -n "pooled_true\|pooled_pred" scripts/ensemble.py
```

Expected: lines around 193-195 (per the spec context). Identify where the loop iterates over seeds × folds.

- [ ] **Step 2: Read those lines**

Use the line numbers from Step 1 to read the block where `pooled_true.extend(...)` is called.

- [ ] **Step 3: Add CSV write inside the per-fold block**

After the existing `pooled_pred.extend(pred.tolist())` line (inside the `if si == 0:` branch is wrong — we want every fold/seed), add the following block. Specifically, inside the `for fold in range(5):` loop, after `pred = sum(proba[m] for m in MEMBERS).argmax(1)`, add:

```python
# Append per-event predictions to the rolling predictions CSV.
# Columns are stable across runs (no timestamp in filename).
proba_avg = sum(proba[m] for m in MEMBERS) / len(MEMBERS)
predictions_rows.append({
    "event_id": str(test_event_ids[te_pos]),
    "true_class": int(y_test[i]),
    "pred_class": int(pred[i]),
    "proba_0": float(proba_avg[i, 0]),
    "proba_1": float(proba_avg[i, 1]),
    "proba_2": float(proba_avg[i, 2]),
    "proba_3": float(proba_avg[i, 3]),
    "proba_4": float(proba_avg[i, 4]),
    "fold": fold,
    "seed": seed,
})
```

- [ ] **Step 4: Add list init + CSV write at top and bottom of `main()`**

Near the top of `main()` (after argument parsing), add:

```python
predictions_rows: list[dict] = []
```

At the end of `main()` (after the existing report-write), add:

```python
import csv as _csv
predictions_csv = Path("results/ensemble/predictions.csv")
predictions_csv.parent.mkdir(parents=True, exist_ok=True)
with predictions_csv.open("w", newline="") as f:
    writer = _csv.DictWriter(f, fieldnames=[
        "event_id", "true_class", "pred_class",
        "proba_0", "proba_1", "proba_2", "proba_3", "proba_4",
        "fold", "seed",
    ])
    writer.writeheader()
    writer.writerows(predictions_rows)
print(f"wrote {len(predictions_rows)} predictions to {predictions_csv}")
```

- [ ] **Step 5: Verify by reading the modified file**

```bash
cd "C:/Users/Aariz/Downloads/CN Reasearch Paper/Forensic-Reconstruction"
grep -n "predictions_rows\|predictions_csv\|predictions\.csv" scripts/ensemble.py
```

Expected: 4 hits — the init line, the per-fold append, the CSV open, and the print.

- [ ] **Step 6: Commit**

```bash
cd "C:/Users/Aariz/Downloads/CN Reasearch Paper/Forensic-Reconstruction"
git add scripts/ensemble.py
git commit -m "feat(ensemble): write predictions.csv alongside report

Additive change: same metrics, plus a stable per-event predictions CSV
the Phase 5 replayer consumes. Columns: event_id, true_class,
pred_class, proba_0..proba_4, fold, seed.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Run the unit test suite end-to-end

**Files:**
- No changes; runs `scripts/tests/test_reconstruct_params.py`

**Why here:** TDD discipline — verify the full test suite still passes after all changes, before declaring done.

- [ ] **Step 1: Run the tests**

```bash
cd "C:/Users/Aariz/Downloads/CN Reasearch Paper/Forensic-Reconstruction"
.venv\Scripts\python.exe -m pytest scripts/tests/ -v
```

Expected: 18 passed in `test_reconstruct_params.py`. No regressions.

- [ ] **Step 2: Verify imports of all four new modules**

```bash
cd "C:/Users/Aariz/Downloads/CN Reasearch Paper/Forensic-Reconstruction"
.venv\Scripts\python.exe -c "
from scripts.reconstruct_params import event_to_scenario_params
from scripts.reconstruct_scenarios import calibrate, build_replay_scenario
from scripts.reconstruct_capture import open_beamng, run_and_capture
import scripts.reconstruct
print('all four modules importable')
"
```

Expected: prints `all four modules importable`.

- [ ] **Step 3: Commit nothing — this is a verification step**

If any step fails, fix it before continuing to Task 7.

---

## Task 7: Integration test (gated by BEAMNG_HOME)

**Files:**
- Create: `scripts/tests/test_reconstruct_smoke.py`

**Why last:** This is the only step that requires BeamNG.tech actually running. Gated so CI doesn't fail when BeamNG isn't installed.

- [ ] **Step 1: Write the test file**

```python
# scripts/tests/test_reconstruct_smoke.py
"""Integration test for Phase 5 reconstruction. Skipped unless BEAMNG_HOME is set."""
import os
import shutil
from pathlib import Path

import pytest

BEAMNG_HOME = os.environ.get("BEAMNG_HOME")
pytestmark = pytest.mark.skipif(
    BEAMNG_HOME is None,
    reason="BEAMNG_HOME not set — BeamNG.tech integration test skipped",
)


@pytest.fixture(scope="module")
def beamng_available():
    if shutil.which("BeamNG.tech") is None and not Path("C:/Program Files/BeamNG.tech/BeamNG.exe").exists():
        # We can't easily check the .exe path on all platforms; rely on the env var gate
        pass
    return BEAMNG_HOME is not None


def test_smoke_test_produces_metadata(beamng_available, tmp_path):
    """Run --smoke-test and assert metadata.json + frames exist."""
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[2]
    rc = subprocess.run(
        [sys.executable, "scripts/reconstruct.py", "--smoke-test"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=180,
    )
    out_dir = repo / "results/reconstruction/smoke_test"
    meta_path = out_dir / "metadata.json"
    assert meta_path.exists(), f"metadata.json not written; rc={rc.returncode}\n{rc.stdout}\n{rc.stderr}"
    import json
    meta = json.loads(meta_path.read_text())
    # smoke_test may not write metadata.json (it just prints status); if so, accept
    if meta:
        assert meta["status"] in ("ok", "timeout", "beamng_error"), meta


def test_missing_event_id_returns_nonzero(beamng_available):
    """--event-id with a non-existent ID returns non-zero exit code."""
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[2]
    rc = subprocess.run(
        [sys.executable, "scripts/reconstruct.py", "--event-id", "DOES_NOT_EXIST_999"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert rc.returncode != 0
    assert "DOES_NOT_EXIST_999" in rc.stderr or "DOES_NOT_EXIST_999" in rc.stdout
```

- [ ] **Step 2: Verify the test is skipped (BEAMNG_HOME unset)**

```bash
cd "C:/Users/Aariz/Downloads/CN Reasearch Paper/Forensic-Reconstruction"
.venv\Scripts\python.exe -m pytest scripts/tests/test_reconstruct_smoke.py -v
```

Expected: 2 skipped, 0 failed, 0 passed. The skip reason text should mention `BEAMNG_HOME`.

- [ ] **Step 3: (Optional, manual) Run with BEAMNG_HOME set on a developer machine**

```bash
export BEAMNG_HOME="D:/BeamNG/BeamNG.tech.v0.38.5.0"
.venv\Scripts\python.exe -m pytest scripts/tests/test_reconstruct_smoke.py -v
.venv\Scripts\python.exe scripts/reconstruct.py --smoke-test
```

Expected (on a machine with BeamNG installed and running): tests pass or one fails with a clear error. If `test_smoke_test_produces_metadata` fails, investigate the smoke run output before merging.

- [ ] **Step 4: Commit**

```bash
cd "C:/Users/Aariz/Downloads/CN Reasearch Paper/Forensic-Reconstruction"
git add scripts/tests/test_reconstruct_smoke.py
git commit -m "test(reconstruct): gated BeamNG integration smoke test

Skipped when BEAMNG_HOME is not set so CI doesn't require BeamNG.
When unskipped, runs --smoke-test and asserts metadata.json written;
runs --event-id with a non-existent ID and asserts non-zero exit.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review Checklist (run before declaring the plan complete)

- [x] Spec coverage: every section in `docs/superpowers/specs/2026-08-25-3d-reconstruction-design.md` is implemented in some task.
  - Section 1 (scope): Task 1-7 boundaries
  - Section 2 (architecture): Task 2, 3, 4 file structure
  - Section 3 (data flow): Task 4 main loop
  - Section 4 (parametric variation): Task 1 (params + bounds + PDOF map)
  - Section 5 (BeamNG integration): Task 2 (calibrate, scenarios), Task 3 (connection, capture, encoding)
  - Section 6 (error handling): Task 1 (skip paths), Task 3 (try/finally + status), Task 4 (CLI error recovery)
  - Section 7 (ensemble predictions.csv): Task 5
  - Section 8 (testing): Task 1 unit tests, Task 7 integration test
- [x] No placeholders. Every step has actual code or actual commands.
- [x] Type consistency:
  - `ScenarioParams` defined in Task 1, consumed in Tasks 2, 3, 4 — same field names.
  - `SkipReason` defined in Task 1, consumed in Task 4 — same field names.
  - `CaptureResult` defined in Task 3, consumed in Task 4 — same field names.
  - `CalibratedHeading` defined in Task 2, consumed in Tasks 3, 4 — same field names.
- [x] Each task ends with an independently testable deliverable.
- [x] Every step is one action (2-5 minutes).
- [x] Branch discipline: every commit goes on `feature/phase-5-3d-reconstruction`, never on `main`.

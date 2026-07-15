# NHTSA CISS EDR Data Codes Reference

## Overview

The CISS dataset contains 999 codes throughout categorical fields. NHTSA uses 999 as a standard indicator for "unknown" or "not recorded" data. A critical distinction exists between 999 and 0 in EDR data.

---

## Distinguishing 999 from 0

**999 = Missing / Unknown**
- Indicates data was not documented or the sensor did not capture the value
- Example: `LIGHTCOND = 999` denotes unknown lighting conditions, not darkness
- Should not be treated as valid measurement data

**0 = Actual Measurement**
- Represents a real sensor reading or valid state
- Example: `steering_deg = 0` indicates steering wheel was centered (no turning input)
- Example: `brake_active = 0` indicates brake pedal was not engaged

---

---

## Field Codes and Mappings

### CRASHTYPE — Classification of First Harmful Event

CRASHTYPE codes describe the nature of the collision. The following mapping applies to the 5-class crash classification system:

| Code Range | Crash Type | Classification |
|---|---|---|
| 1–4 | Rear-end collision | Class 0: Rear-end |
| 5 | Head-on collision | Class 2: Head-on |
| 6–10 | Angle/intersection collision | Class 1: Angle |
| 11–12 | Sideswipe collision | Class 4: Sideswipe |
| 55–69 | Single-vehicle (rollover, fixed object impact, run-off-road) | Class 3: Single-vehicle |
| 998, 999 | Unclassifiable / Unknown | Excluded from training set |

Cases with unmapped CRASHTYPE values (998, 999) are excluded from the training dataset, as they lack a valid classification label.

---

### SURFCOND — Road Surface Condition

Road surface condition directly impacts friction coefficient, which is essential for physics-based validation of reconstructed crash scenarios.

| Code | Surface Type | Typical Friction Coefficient (μ) |
|---|---|---|
| 1 | Dry pavement | 0.7–0.9 |
| 2 | Wet surface | 0.5–0.7 |
| 3 | Snow/slush | 0.2–0.5 |
| 4 | Ice | 0.1–0.3 |
| 5–7 | Gravel/dirt/sand | 0.4–0.6 |
| 8 | Oil/contaminated | < 0.3 |
| 99, 999 | Unknown | Cannot estimate friction coefficient |

The friction coefficient is used in the physics validation phase to compute expected braking distance from recorded deceleration. If SURFCOND is unknown (999), the physics consistency check cannot validate kinematic parameters against recorded delta-V data.

---

### LIGHTCOND — Ambient Lighting Conditions

| Code | Condition | Visibility Impact |
|---|---|---|
| 1 | Daylight | Optimal visibility |
| 2 | Dusk/dawn | Reduced visibility, transition lighting |
| 3 | Dark with street lights | Moderate visibility with illumination |
| 4 | Dark, no street lights | Poor visibility |
| 5 | Dark, non-functional street lights | No artificial illumination |
| 99, 999 | Unknown | Data not available |

Lighting conditions serve as a contextual feature for post-hoc analysis of model performance. Crashes occurring in low-light conditions may exhibit different pre-crash kinematic signatures compared to daylight crashes.

---

### WEATHER — Atmospheric Conditions

| Code | Condition | Physical Characteristics |
|---|---|---|
| 1 | Clear/sunny | Normal atmospheric conditions |
| 2 | Rain | Reduced friction on road surface |
| 3 | Sleet/freezing rain | Icy surface formation |
| 4 | Snow | Low friction, variable surface properties |
| 5 | Fog | Reduced visibility |
| 6 | Strong winds | Potential cross-wind forces |
| 7 | Severe crosswinds | Significant lateral forces on vehicles |
| 99, 999 | Unknown | Data not available |

Weather conditions often correlate with SURFCOND but provide independent information. Rain combined with an already-wet road (SURFCOND=2) presents different dynamics than rain on dry pavement (SURFCOND=1).

---

### MANEUVER — Pre-Crash Vehicle Maneuver

| Code | Vehicle Action |
|---|---|
| 1 | Traveling straight |
| 2 | Accelerating |
| 3 | Decelerating/Braking |
| 4 | Turning left |
| 5 | Turning right |
| 6 | Backing up |
| 7 | Changing lanes |
| 8 | Merging into traffic |
| 9 | Overtaking |
| 10 | Parked/Stationary |
| 99, 999 | Unknown/unrecorded |

While MANEUVER may not always be directly available in EDR data, it can be inferred from kinematic parameters. For example, acceleration values and throttle input indicate acceleration (code 2), while brake_active and negative accel_long values indicate braking (code 3). The Transformer model learns these associations implicitly from raw EDR sensor values.

---

### Delta-V Fields — Impact Velocity Change

| Field | Definition | Units | Use Case |
|---|---|---|---|
| DVTOTAL | Total magnitude of velocity change | km/h | Overall crash severity assessment |
| DVLONG | Longitudinal velocity change | km/h | Front/rear impact intensity |
| DVLAT | Lateral velocity change | km/h | Side impact intensity |

A value of 999 in delta-V fields indicates the impact was not measured (sensor unavailable or EDR failure). Delta-V measurements serve as ground truth for the physics validation layer, which computes expected impact velocity change from reconstructed pre-crash kinematics and compares it to the recorded EDR delta-V values.

---

---

## Data Handling and Quality Considerations

### Treatment of 999 Codes

In the current cleaning pipeline, 999 values are retained in the dataset. This approach assumes the model can learn to treat 999 as a distinct missing-data category through the presence of explicit missing indicators. Alternative approaches include:

- **Option A (Current):** Retain 999 values with accompanying missing-data flags
- **Option B:** Replace all 999 values with NaN and rely on missing-indicator columns
- **Option C:** Drop all cases containing 999 values (risks losing legitimate data)

### Data Quality Tiers

Cases differ significantly in completeness and data quality:

**High-Quality Cases (< 15% missing fields)**
- SURFCOND, LIGHTCOND, WEATHER codes are documented
- CRASHTYPE is clearly mapped (1–69)
- Complete or near-complete EDR timeseries
- Expected model confidence: High

**Low-Quality Cases (> 30% missing fields)**
- SURFCOND or LIGHTCOND = 999
- DVTOTAL = 999 or unavailable
- Significant gaps in EDR timeseries
- CRASHTYPE unmapped (998, 999)
- Expected model confidence: Low

The pipeline should separately report model performance metrics on high-quality vs. low-quality subsets to provide realistic performance expectations.

---

## Field Mapping to Unified Intermediate Representation (UIR)

The following mapping standardizes fields from raw CISS EDR into the unified representation used for model training:

| UIR Column | CISS Source | Data Type | Processing |
|---|---|---|---|
| speed_kmh | EDRPRECRASH 1010 | Continuous | Cubic interpolation to 10 Hz |
| accel_long | EDRPRECRASH 1020 | Continuous | Cubic interpolation to 10 Hz |
| accel_lat | EDRPRECRASH 1030 | Continuous | Cubic interpolation to 10 Hz |
| brake_active | EDRPRECRASH 1050 | Binary | Forward-fill to 10 Hz; impute from accel_long if missing |
| throttle_pct | EDRPRECRASH 1040 | Continuous | Cubic interpolation; mark missing if unavailable |
| steering_deg | EDRPRECRASH 1060 | Continuous | Cubic interpolation; mark missing if unavailable |
| crash_class | CRASHTYPE | Categorical | Map codes 1–69 to classes 0–4; exclude unmapped |
| delta_v_long | EDRSUMM DVLONG | Continuous | Used for physics validation layer only |
| surface_condition | SURFCOND | Categorical | Retained for context analysis |
| lighting_condition | LIGHTCOND | Categorical | Retained for context analysis |

---

## Summary

Understanding NHTSA CISS data codes is essential for correct interpretation of EDR measurements and accurate crash reconstruction:

1. **Missing data (999)** is distinct from measured zero values — distinguishing between them is critical for feature engineering and model training.

2. **CRASHTYPE** provides ground-truth labels for supervised classification. Unmapped codes (998, 999) must be excluded from training.

3. **SURFCOND** enables physics-based validation through friction coefficient estimation. Missing SURFCOND values (999) reduce the confidence of physics checks.

4. **LIGHTCOND and WEATHER** provide contextual information for post-hoc analysis and may correlate with pre-crash kinematic signatures.

5. **MANEUVER** can be inferred from kinematic parameters; explicit MANEUVER codes serve as supplementary validation signals.

6. **Delta-V** measurements serve as reference ground truth for validating reconstructed impact severity. Missing delta-V data (999) limits physics-layer validation capability.

7. **Data quality varies significantly** across cases. Reporting separate performance metrics for high-quality and low-quality subsets provides transparent evaluation of model capabilities.


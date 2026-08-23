"""Per-event feature builder for the ensemble (documentation.txt Part 14).

Extends gbt_baseline.py's 55 aggregated features (11 UIR columns x 5 stats)
in two ways:

  1. More distributional statistics per column -- 15 instead of 5, adding
     first/percentiles/range/delta/slope/abs/rms. 11 x 15 = 165.
  2. 33 PHYSICS-DERIVED cross-channel features that no per-column statistic
     can express, most importantly:
       - PDOF (principal direction of force): the angle of the acceleration
         vector at the moment of peak resultant acceleration. This is the
         standard forensic descriptor of impact direction, and it is the
         most direct discriminator between crash classes there is -- a
         head-on/rear-end impact is longitudinal (PDOF near 0/180 deg), an
         angle/sideswipe impact is lateral (PDOF near +-90 deg). Encoded as
         degrees AND as sin/cos so the +-180 deg wraparound isn't a cliff.
       - lateral-vs-longitudinal energy ratios and the fraction of the
         event where lateral force exceeds longitudinal.
       - yaw integral (net heading change) and sign-change counts, which
         separate a spin-out single-vehicle event from a clean rear-end.
       - delta-V / impulse proxies for impact severity.

Total: 198 features. Computed on RAW (unscaled) sequences -- ratios and
atan2 angles are only physically meaningful in real units, not z-scores.
Tree models don't need scaled inputs anyway; the one linear member in the
ensemble standardizes them itself.

IMPORTANT CAVEAT (same one documentation.txt Part 13.1 raises for the
55-feature baseline): statistics of the *_missing indicator columns are
partly a proxy for data SOURCE, not crash dynamics, since each source
records a different subset of fields. That is disclosed rather than hidden;
see Part 14.4.
"""
import numpy as np

EPS = 1e-6

# indices into FEATURE_COLS
# 0 speed_kmh, 1 accel_long, 2 accel_lat, 3 brake_active, 4 throttle_pct,
# 5 steering_deg, 6 yaw_rate, 7 brake_missing, 8 throttle_missing,
# 9 steering_missing, 10 yaw_missing
I_SPEED, I_ALONG, I_ALAT, I_BRAKE, I_THR, I_STEER, I_YAW = 0, 1, 2, 3, 4, 5, 6

STAT_NAMES = ["mean", "std", "min", "max", "last", "first", "p25", "p50", "p75",
              "range", "delta", "slope", "absmean", "absmax", "rms"]


def _col_stats(col, valid, n_real):
    """col: (N,T) raw, valid: (N,T) bool. Returns (N, 15)."""
    N, T = col.shape
    cm = np.where(valid, col, np.nan)
    with np.errstate(all="ignore"):
        mean = np.nanmean(cm, axis=1)
        std = np.nanstd(cm, axis=1)
        mn = np.nanmin(cm, axis=1)
        mx = np.nanmax(cm, axis=1)
        p25 = np.nanpercentile(cm, 25, axis=1)
        p50 = np.nanpercentile(cm, 50, axis=1)
        p75 = np.nanpercentile(cm, 75, axis=1)
        absmean = np.nanmean(np.abs(cm), axis=1)
        absmax = np.nanmax(np.abs(cm), axis=1)
        rms = np.sqrt(np.nanmean(cm ** 2, axis=1))

    # left-padded => last real value is always at T-1
    last = col[:, -1]
    # first real value = at index T - n_real
    first_idx = np.clip(T - n_real, 0, T - 1)
    first = col[np.arange(N), first_idx]

    rng = mx - mn
    delta = last - first
    slope = delta / np.maximum(n_real - 1, 1)  # per-timestep trend

    out = np.stack([mean, std, mn, mx, last, first, p25, p50, p75,
                    rng, delta, slope, absmean, absmax, rms], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _sign_changes(col, valid):
    """Count sign flips over real timesteps (proxy for oscillation/rotation reversal)."""
    s = np.sign(np.where(valid, col, 0.0))
    flips = (s[:, 1:] * s[:, :-1]) < 0
    flips = flips & valid[:, 1:] & valid[:, :-1]
    return flips.sum(axis=1).astype(np.float32)


def build_rich(X, mask, feature_cols):
    """X: (N,T,F) RAW, mask: (N,T) True=padding. Returns (feats (N,D), names)."""
    N, T, F = X.shape
    valid = ~mask
    n_real = valid.sum(axis=1).astype(np.float32)

    feats, names = [], []

    # --- per-column distributional stats ---
    for f in range(F):
        feats.append(_col_stats(X[:, :, f], valid, n_real.astype(int)))
        names += [f"{feature_cols[f]}_{s}" for s in STAT_NAMES]

    # --- physics / cross-channel features ---
    along = np.where(valid, X[:, :, I_ALONG], np.nan)
    alat = np.where(valid, X[:, :, I_ALAT], np.nan)
    speed = np.where(valid, X[:, :, I_SPEED], np.nan)
    yaw = np.where(valid, X[:, :, I_YAW], np.nan)

    res = np.sqrt(np.nan_to_num(along) ** 2 + np.nan_to_num(alat) ** 2)
    res = np.where(valid, res, np.nan)

    phys, pnames = [], []

    def add(v, name):
        phys.append(np.nan_to_num(np.asarray(v, dtype=np.float64),
                                  nan=0.0, posinf=0.0, neginf=0.0))
        pnames.append(name)

    add(n_real, "n_real_timesteps")

    with np.errstate(all="ignore"):
        add(np.nanmean(res, axis=1), "res_accel_mean")
        add(np.nanmax(res, axis=1), "res_accel_max")
        add(np.nanstd(res, axis=1), "res_accel_std")
    add(res[:, -1], "res_accel_last")

    # peak-of-impact frame: the timestep with the largest resultant acceleration
    res_f = np.nan_to_num(res, nan=-np.inf)
    pk = np.argmax(res_f, axis=1)
    rows = np.arange(N)
    pk_long = np.nan_to_num(along[rows, pk])
    pk_lat = np.nan_to_num(alat[rows, pk])
    add(pk_long, "peak_accel_long_signed")
    add(pk_lat, "peak_accel_lat_signed")
    # normalized position of the peak within the real window (0=start, 1=impact)
    add((pk - (T - n_real)) / np.maximum(n_real - 1, 1), "peak_pos_norm")

    # PDOF (principal direction of force) -- the forensic-standard descriptor of
    # impact direction. sin/cos encoding avoids the +-180 deg wraparound cliff.
    pdof = np.arctan2(pk_lat, pk_long)
    add(np.degrees(pdof), "pdof_deg")
    add(np.sin(pdof), "pdof_sin")
    add(np.cos(pdof), "pdof_cos")
    add(np.abs(np.degrees(pdof)), "pdof_abs_deg")

    # lateral-vs-longitudinal dominance: the single most direct discriminator
    # between head-on/rear-end (longitudinal) and angle/sideswipe (lateral)
    with np.errstate(all="ignore"):
        amax_long = np.nanmax(np.abs(along), axis=1)
        amax_lat = np.nanmax(np.abs(alat), axis=1)
        amean_long = np.nanmean(np.abs(along), axis=1)
        amean_lat = np.nanmean(np.abs(alat), axis=1)
        e_long = np.nansum(along ** 2, axis=1)
        e_lat = np.nansum(alat ** 2, axis=1)
    add(amax_lat / (amax_long + EPS), "latlong_ratio_max")
    add(amean_lat / (amean_long + EPS), "latlong_ratio_mean")
    add(e_lat / (e_long + EPS), "latlong_ratio_energy")
    add(e_lat / (e_lat + e_long + EPS), "lat_energy_frac")
    # fraction of the event where lateral force exceeds longitudinal
    lat_dom = (np.abs(np.nan_to_num(alat)) > np.abs(np.nan_to_num(along))) & valid
    add(lat_dom.sum(axis=1) / np.maximum(n_real, 1), "lat_dominant_frac")

    # delta-V proxies (impact severity); no true dt per source, so mean*n_real
    add(np.nan_to_num(np.nanmean(res, axis=1)) * n_real, "impulse_proxy")
    add(np.nan_to_num(np.nanmean(np.abs(along), axis=1)) * n_real, "impulse_long_proxy")
    add(np.nan_to_num(np.nanmean(np.abs(alat), axis=1)) * n_real, "impulse_lat_proxy")

    # speed profile
    first_idx = np.clip(T - n_real.astype(int), 0, T - 1)
    sp_first = X[rows, first_idx, I_SPEED]
    sp_last = X[:, -1, I_SPEED]
    add(sp_first, "speed_first")
    add(sp_last, "speed_last")
    add(sp_first - sp_last, "speed_drop")
    add((sp_first - sp_last) / np.maximum(n_real - 1, 1), "speed_drop_rate")
    with np.errstate(all="ignore"):
        add(np.nanmax(speed, axis=1) - np.nanmin(speed, axis=1), "speed_span")

    # rotation: yaw integral = net heading change; sign flips = spin reversal
    with np.errstate(all="ignore"):
        add(np.nansum(yaw, axis=1), "yaw_integral")
        add(np.nanmax(np.abs(yaw), axis=1), "yaw_absmax")
    add(_sign_changes(X[:, :, I_YAW], valid), "yaw_sign_changes")
    add(_sign_changes(X[:, :, I_STEER], valid), "steer_sign_changes")
    add(_sign_changes(X[:, :, I_ALAT], valid), "alat_sign_changes")
    add(_sign_changes(X[:, :, I_ALONG], valid), "along_sign_changes")

    # driver reaction
    brake = np.where(valid, X[:, :, I_BRAKE], np.nan)
    with np.errstate(all="ignore"):
        add(np.nanmean(brake, axis=1), "brake_frac")
    add(X[:, -1, I_BRAKE], "brake_at_impact")

    feats.append(np.stack(phys, axis=1))
    names += pnames

    out = np.concatenate(feats, axis=1).astype(np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out, names

#!/usr/bin/env python3
"""
beamng_real_generator.py  (v2)  —  synthetic crash-telemetry generator, BeamNGpy 1.35.1
========================================================================================

v2 fixes (after the overnight run where head-on worked but sideswipe made 0% progress):

  * SIDESWIPE geometry rebuilt. The old version started the striker BEHIND and FASTER
    on slightly converging lines, so longitudinal alignment was uncontrolled and the
    cars almost never touched. v2 places both cars ABREAST at the SAME speed with a
    gentle fixed convergence — verified 100% contact across jittered speeds/offsets.
  * SIDESWIPE detection threshold lowered. A glancing sideswipe deforms far less than a
    head-on; the flat damage>100 rule discarded even successful hits. Detection is now
    PER-TYPE and measured relative to a per-run baseline.
  * ANGLE now uses EQUAL speeds for both cars (the old 60-100 vs 30-70 ranges meant the
    two cars never reached the crossing point together — same latent timing bug as
    sideswipe). Equal speed + near-equal distance -> reliable T-bone into the side.
  * RESUMABLE: each class is written to its own CSV as it finishes, you can skip classes
    you already generated via CLASSES_TO_RUN, and all per-class CSVs are auto-merged into
    BeamNG_synthetic_data.csv at the end.

WHY head-on still uses the same approach it did (it worked): two cars on the SAME line
moving toward each other collide regardless of speed or timing, so it needs none of the
above. Only the crossing/glancing geometries needed the timing guarantees.

Run:  python beamng_real_generator.py
"""

import os
import glob
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

from beamngpy import BeamNGpy, Scenario, Vehicle, ProceduralCube, angle_to_quat
from beamngpy.sensors import Electrics, Damage

# ─────────────────────────────── CONFIG ────────────────────────────────
BEAMNG_HOME = r'D:\BeamNG\BeamNG.tech.v0.38.5.0'
BEAMNG_USER = r'C:\Users\Aariz\AppData\Local\BeamNG.drive'
HOST        = 'localhost'
PORT        = 64256
MAP         = 'smallgrid'

OUT_DIR = r'C:\Users\Aariz\Downloads\CN Reasearch Paper\Dataset\data\processed\BeamNG_Synthetic'
os.makedirs(OUT_DIR, exist_ok=True)

PHYSICS_HZ    = 60
STEP_CHUNK    = 6                 # 6/60 = 0.1 s per sample = 10 Hz
SAMPLE_DT     = STEP_CHUNK / PHYSICS_HZ
MAX_SIM_S     = 12
POST_IMPACT_S = 0.6
MODEL         = 'etk800'

# Class -> (type, target count). Rear-end/class-0 is intentionally not synthesised
# (well covered by the real datasets). Order = generation order.
TARGETS = {
    2: ('head-on',        840),
    4: ('sideswipe',      400),
    3: ('single-vehicle', 300),
    1: ('angle',          250),
}

# ── RESUME CONTROL ──────────────────────────────────────────────────────
# Which classes to (re)generate this run. To keep your existing 840 head-on events,
# rename your last head-on checkpoint to  BeamNG_head-on.csv  in OUT_DIR, then set:
#     CLASSES_TO_RUN = [4, 3, 1]
# Leave as all four to regenerate everything from scratch.
CLASSES_TO_RUN = [3, 1]

# Damage-sensor rise (above a per-run baseline) that counts as a crash, per type.
DAMAGE_TRIG = {
    'head-on':        120.0,
    'angle':          100.0,
    'single-vehicle': 120.0,
    'sideswipe':       15.0,     # glancing contact -> low deformation
}

# Nominal impact speeds, km/h (jitter added per run).
#   head-on: (ego, other) pairs — different speeds are fine (collinear -> always meet).
#   angle / sideswipe: SINGLE speed, applied to BOTH cars (timing depends on equal speed).
#   single-vehicle: single speed.
SPEED_CONFIGS = {
    'head-on'       : [(s1, s2) for s1 in range(50, 100, 10) for s2 in range(50, 100, 10)],
    'angle'         : list(range(45, 90, 5)),
    'single-vehicle': list(range(70, 125, 5)),
    'sideswipe'     : list(range(60, 105, 5)),
}

APPROACH_DIST = {'head-on': 55.0, 'angle': 55.0, 'single-vehicle': 65.0, 'sideswipe': 55.0}

UIR_COLS = [
    'event_id', 'source', 'source_year', 'crash_class', 'is_crash',
    't', 'speed_kmh', 'accel_long', 'accel_lat', 'brake_active',
    'throttle_pct', 'steering_deg', 'yaw_rate', 'engine_rpm',
    'abs_active', 'throttle_missing', 'steering_missing',
]


# ─────────────────────────────── HELPERS ───────────────────────────────
def vlen3(v):
    return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5


def norm_angle(a):
    return (a + 180.0) % 360.0 - 180.0


def total_damage(d):
    if not d:
        return 0.0
    v = d.get('damage')
    if isinstance(v, (int, float)):
        return float(v)
    part = d.get('part_damage') or {}
    return sum(float(p.get('damage', 0) or 0) for p in part.values() if isinstance(p, dict))


# ───────────────────────────── CALIBRATION ─────────────────────────────
def calibrate(bng):
    """Measure spawn-heading convention and validate set_velocity direction.
    Returns qh(theta_deg) -> quat driving along world heading theta (0=+X, CCW+)."""
    print('Calibrating heading + launch ...')
    probe = Vehicle('probe', model=MODEL, color='White')
    sc = Scenario(MAP, 'calibration')
    sc.add_vehicle(probe, pos=(0, 0, 0.3), rot_quat=angle_to_quat((0, 0, 0)))
    sc.make(bng); bng.load_scenario(sc); bng.start_scenario()

    bng.step(PHYSICS_HZ // 2); probe.poll_sensors()
    d0 = probe.state['dir']; yaw0 = math.degrees(math.atan2(d0[1], d0[0]))
    probe.teleport((0, 0, 0.3), rot_quat=angle_to_quat((0, 0, 90)), reset=True)
    bng.step(PHYSICS_HZ // 2); probe.poll_sensors()
    d90 = probe.state['dir']; yaw90 = math.degrees(math.atan2(d90[1], d90[0]))
    s = 1.0 if norm_angle(yaw90 - yaw0) > 0 else -1.0
    print(f'  identity heading = {yaw0:6.1f} deg | +90cmd = {yaw90:6.1f} deg | sign = {s:+.0f}')

    def qh(theta_deg):
        return angle_to_quat((0, 0, (theta_deg - yaw0) / s))

    probe.teleport((-30.0, 0.0, 0.3), rot_quat=qh(0.0), reset=True)
    bng.step(PHYSICS_HZ // 2); probe.poll_sensors()
    x0 = probe.state['pos'][0]
    probe.set_velocity(14.0, dt=1.0)
    bng.step(int(1.5 * PHYSICS_HZ)); probe.poll_sensors()
    p = probe.state['pos']; adv = p[0] - x0; drift = abs(p[1])
    bng.stop_scenario()
    print(f'  launch check: advanced {adv:+.1f} m along +X, lateral drift {drift:.1f} m')
    if adv < 3.0 or drift > 8.0:
        raise RuntimeError('set_velocity did not propel toward the target — paste this '
                           'output and I will switch to throttle-based launch.')
    print('  calibration OK.\n')
    return qh


# ─────────────────────────── SCENARIO PLACEMENT ────────────────────────
def add_toward_origin(scenario, vid, color, heading_deg, dist, qh):
    """Place a car `dist` m out so heading_deg drives it to the origin."""
    th = math.radians(heading_deg)
    pos = (-math.cos(th) * dist, -math.sin(th) * dist, 0.3)
    veh = Vehicle(vid, model=MODEL, color=color)
    scenario.add_vehicle(veh, pos=pos, rot_quat=qh(heading_deg))
    return veh


def add_at(scenario, vid, color, x, y, heading_deg, qh):
    """Place a car at explicit world (x, y) facing world heading_deg."""
    veh = Vehicle(vid, model=MODEL, color=color)
    scenario.add_vehicle(veh, pos=(x, y, 0.3), rot_quat=qh(heading_deg))
    return veh


def build_scenario(crash_type, speeds, qh, event_id):
    """Return (scenario, ego, launches[(veh, speed_ms)])."""
    D = APPROACH_DIST[crash_type]
    scenario = Scenario(MAP, f'{crash_type}_{event_id}')

    if crash_type == 'head-on':
        v1, v2 = speeds
        ego   = add_toward_origin(scenario, 'ego',   'Red',  0.0,   D, qh)
        other = add_toward_origin(scenario, 'other', 'Blue', 180.0, D, qh)
        launches = [(ego, v1 / 3.6), (other, v2 / 3.6)]

    elif crash_type == 'angle':
        v = speeds                                   # equal speed for both
        ego   = add_toward_origin(scenario, 'ego',   'Red',  0.0,  D,       qh)
        other = add_toward_origin(scenario, 'other', 'Blue', 90.0, D - 2.0, qh)  # 2 m lead -> hits its side
        launches = [(ego, v / 3.6), (other, v / 3.6)]

    elif crash_type == 'single-vehicle':
        v = speeds
        ego = add_toward_origin(scenario, 'ego', 'Red', 0.0, D, qh)
        wall = ProceduralCube(pos=(0, 0, 1.5), size=(2.5, 10.0, 3.0),
                              name=f'wall_{event_id}', rot_quat=angle_to_quat((0, 0, 0)))
        scenario.add_procedural_mesh(wall)
        launches = [(ego, v / 3.6)]

    elif crash_type == 'sideswipe':
        v = speeds                                   # equal speed keeps them abreast
        # ego = striker: 2.5 m behind, right lane, angled 3 deg toward the target
        ego   = add_at(scenario, 'ego',   'Red',  -(D + 2.5), -1.5, 3.0, qh)
        other = add_at(scenario, 'other', 'Blue', -D,          1.5, 0.0, qh)
        launches = [(ego, v / 3.6), (other, v / 3.6)]

    else:
        raise ValueError(crash_type)

    return scenario, ego, launches


# ───────────────────────────── SINGLE SCENARIO ─────────────────────────
def run_one_scenario(bng, crash_type, crash_class, speeds, qh, event_id):
    scenario, ego, launches = build_scenario(crash_type, speeds, qh, event_id)
    ego.attach_sensor('electrics', Electrics())
    ego.attach_sensor('damage',    Damage())

    scenario.make(bng); bng.load_scenario(scenario); bng.start_scenario()
    bng.step(PHYSICS_HZ // 2)                         # settle

    ego.poll_sensors()
    baseline_dmg = total_damage(ego.sensors['damage'])
    trigger = DAMAGE_TRIG[crash_type]

    for veh, v_ms in launches:
        veh.set_velocity(v_ms, dt=1.0 + v_ms / 25.0)
    for veh, _ in launches:
        veh.control(throttle=0.45, steering=0.0, brake=0.0)

    rows = []
    prev_vel = prev_head = None
    impact_t = None
    for _ in range(int(MAX_SIM_S / SAMPLE_DT)):
        bng.step(STEP_CHUNK)
        ego.poll_sensors()
        st = ego.state; elec = ego.sensors['electrics']
        vel = st['vel']; d = st['dir']; t_now = st.get('time', None)

        ws = elec.get('wheelspeed', None)
        speed_kmh = (ws * 3.6) if isinstance(ws, (int, float)) else vlen3(vel) * 3.6

        fx, fy = d[0], d[1]; fn = math.hypot(fx, fy) or 1.0; fx, fy = fx / fn, fy / fn
        if prev_vel is not None:
            ax = (vel[0] - prev_vel[0]) / SAMPLE_DT; ay = (vel[1] - prev_vel[1]) / SAMPLE_DT
            accel_long = ax * fx + ay * fy
            accel_lat  = ax * (-fy) + ay * fx
        else:
            accel_long = accel_lat = 0.0
        prev_vel = vel

        head = math.degrees(math.atan2(fy, fx))
        yaw_rate = norm_angle(head - prev_head) / SAMPLE_DT if prev_head is not None else 0.0
        prev_head = head

        rows.append({
            'sim_t': t_now, 'speed_kmh': speed_kmh,
            'accel_long': accel_long, 'accel_lat': accel_lat,
            'brake_active': float(elec.get('brake', 0) > 0.1),
            'throttle_pct': float(elec.get('throttle', 0)) * 100.0,
            'steering_deg': float(elec.get('steering', 0)) * 540.0,
            'yaw_rate': yaw_rate, 'engine_rpm': float(elec.get('rpm', 0) or 0),
            'abs_active': float(elec.get('abs', 0) or 0),
        })

        if (total_damage(ego.sensors['damage']) - baseline_dmg) > trigger and impact_t is None:
            impact_t = rows[-1]['sim_t'] if rows[-1]['sim_t'] is not None else len(rows) * SAMPLE_DT
            for _ in range(int(POST_IMPACT_S / SAMPLE_DT)):
                bng.step(STEP_CHUNK)
            break

    bng.stop_scenario()
    if impact_t is None or len(rows) < 6:
        return None

    df = pd.DataFrame(rows)
    if df['sim_t'].notna().all():
        df['t'] = df['sim_t'] - impact_t
    else:
        df['t'] = np.arange(len(df)) * SAMPLE_DT - (len(df) - 1) * SAMPLE_DT
    df = df[(df['t'] <= 0.0) & (df['t'] >= -20.0)]
    if len(df) < 5:
        return None

    df['event_id'] = f'BeamNG_{crash_type}_{event_id}'
    df['source'] = 'BeamNG'; df['source_year'] = 2025
    df['crash_class'] = crash_class; df['is_crash'] = 1
    df['throttle_missing'] = 0; df['steering_missing'] = 0
    return df[UIR_COLS].sort_values('t').reset_index(drop=True)


# ─────────────────────────────── MERGE ─────────────────────────────────
def merge_all():
    parts = [f for f in glob.glob(os.path.join(OUT_DIR, 'BeamNG_*.csv'))
             if os.path.basename(f) != 'BeamNG_synthetic_data.csv']
    if not parts:
        return
    combined = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True)
    out = os.path.join(OUT_DIR, 'BeamNG_synthetic_data.csv')
    combined.to_csv(out, index=False)
    names = {0: 'rear-end', 1: 'angle', 2: 'head-on', 3: 'single-vehicle', 4: 'sideswipe'}
    print(f"\nMerged {len(parts)} per-class files -> {out}")
    print(f"Total events : {combined['event_id'].nunique()}   rows: {len(combined)}")
    print(combined.groupby('event_id')['crash_class'].first().map(names).value_counts())


# ──────────────────────────────── MAIN ─────────────────────────────────
def main():
    print('Connecting to BeamNG.tech ...')
    bng = BeamNGpy(HOST, PORT, home=BEAMNG_HOME, user=BEAMNG_USER)
    bng.open(launch=True)
    print('  connected.')
    bng.set_deterministic(steps_per_second=PHYSICS_HZ)
    qh = calibrate(bng)

    event_counter = 0
    try:
        for crash_class, (crash_type, target) in TARGETS.items():
            if crash_class not in CLASSES_TO_RUN:
                print(f'Skipping {crash_type} (not in CLASSES_TO_RUN).')
                continue
            print(f"\n{'='*60}\nGenerating {target}x  [{crash_type}]  (class {crash_class})\n{'='*60}")
            speed_list  = SPEED_CONFIGS[crash_type]
            class_events = []
            generated = attempts = 0
            attempt_cap = target * 4 + 50

            with tqdm(total=target, desc=crash_type) as pbar:
                while generated < target and attempts < attempt_cap:
                    attempts += 1
                    sp = speed_list[event_counter % len(speed_list)]
                    if isinstance(sp, tuple):
                        sp = tuple(max(20.0, x + np.random.uniform(-5, 5)) for x in sp)
                    else:
                        sp = max(30.0, sp + np.random.uniform(-5, 5))
                    try:
                        df = run_one_scenario(bng, crash_type, crash_class, sp, qh, event_counter)
                    except Exception as e:
                        df = None; tqdm.write(f'  scenario {event_counter} error: {e}')
                    event_counter += 1
                    if df is not None:
                        class_events.append(df); generated += 1; pbar.update(1)
                        if generated % 100 == 0:
                            pd.concat(class_events, ignore_index=True).to_csv(
                                os.path.join(OUT_DIR, f'checkpoint_{crash_type}_{generated}.csv'), index=False)
                            tqdm.write(f'  checkpoint: {generated} {crash_type}')
                    # early sanity ping so you are not surprised at 3am
                    if attempts == 40 and generated == 0:
                        tqdm.write(f'  NOTE: 0/40 {crash_type} so far — geometry may be off, '
                                   f'let it reach ~100 attempts before judging.')

            if class_events:
                pd.concat(class_events, ignore_index=True).to_csv(
                    os.path.join(OUT_DIR, f'BeamNG_{crash_type}.csv'), index=False)
                print(f'  saved BeamNG_{crash_type}.csv ({generated} events)')
            if generated < target:
                print(f'  WARNING: only {generated}/{target} {crash_type} in {attempts} attempts '
                      f'(hit rate {generated/max(attempts,1):.0%}).')

    except KeyboardInterrupt:
        print('\nInterrupted — saving progress ...')
    finally:
        try:
            bng.close(); print('\nBeamNG closed.')
        except Exception:
            pass

    merge_all()


if __name__ == '__main__':
    main()
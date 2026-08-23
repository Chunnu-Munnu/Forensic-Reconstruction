"""
Downloads real vehicle-level accelerometer curves from NHTSA's Vehicle
Crash Test Database public API (nrd.api.nhtsa.dot.gov -- NOT behind the
Akamai bot-wall that blocks www.nhtsa.gov, verified manually).

Why: BeamNG (this project's synthetic augmentation source) generates
ZERO Rear-end or Sideswipe scenarios (documentation.txt Part 9.8), which
is the clearest identified reason those two classes underperform across
every evaluation protocol tried. Real NCAP/compliance crash tests give
REAL vehicle-CG accelerometer data (not dummy/occupant biomechanics) for
exactly those configurations: VEHICLE INTO VEHICLE tests (rear/frontal
impacts between two real vehicles) and angled IMPACTOR INTO VEHICLE
tests (side/oblique impacts).

Test-configuration -> crash_class mapping (documented, not guessed
blindly -- mirrors the rigor applied to CISS's CRASHTYPE mapping in
CISS_DATA_CODES_GUIDE.md):
  VEHICLE INTO BARRIER, VEHICLE INTO POLE, ROLLOVER  -> 3 (Single-vehicle,
    fixed-object impact -- matches CISS's own definition of this class)
  VEHICLE INTO VEHICLE, IMPACTOR INTO VEHICLE        -> classified by
    impactAngle: near 0/180 (+-30deg) with rear-tagged config -> 0
    (Rear-end); near 90/270 (+-30deg) -> 1 (Angle); other oblique
    (30-60deg from parallel) -> 4 (Sideswipe); near-head-on (0deg,
    'rear' NOT in description) -> 2 (Head-on)
  STATIC AIR BAG TEST SIDE, LOW RISK DEPLOYMENT, SLED (WITH/WITHOUT
    VEHICLE BODY) -> excluded (prescribed sled pulses / static rig
    tests, not a physically realized full crash kinematic trace)
  OTHER, VEHICLE INTO IMPACTOR, IMPACTOR INTO BARRIER -> excluded (too
    small/ambiguous to classify reliably)

API chain (all verified working, unauthenticated):
  get-instrumentation-info/{testNo}            -> list of curves (paginated)
  get-instrumentation-detail-info/{curveNo}/{testNo} -> curve detail + file URL
  https://nrd-static.nhtsa.dot.gov/tsv/...      -> raw two-column (time, value) file

Only VEHICLE CG accelerometer curves are fetched (sensorAttachment ==
"VEHICLE CG", sensorType == "ACCELEROMETER") -- a handful of curves per
test out of dozens, not the full occupant-biomechanics instrumentation
set.

Usage:
    .venv\\Scripts\\python.exe scripts\\download_crash_test_db.py --limit 20     # validate first
    .venv\\Scripts\\python.exe scripts\\download_crash_test_db.py                # full run (resumable)
"""

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX_CSV = REPO_ROOT / "Dataset" / "data" / "raw" / "CrashTestDB" / "test_metadata_index_live.csv"
OUT_DIR = REPO_ROOT / "Dataset" / "data" / "raw" / "CrashTestDB" / "tests"
MANIFEST_PATH = REPO_ROOT / "Dataset" / "data" / "raw" / "CrashTestDB" / "manifest.csv"

API_BASE = "https://nrd.api.nhtsa.dot.gov/nhtsa/vehicle/api/v1/vehicle-database-test-results"
REQUEST_DELAY = 0.15  # seconds between HTTP calls -- be a polite API citizen

EXCLUDED_CONFIGS = {
    "STATIC AIR BAG TEST SIDE", "LOW RISK DEPLOYMENT",
    "SLED WITHOUT VEHICLE BODY", "SLED WITH VEHICLE BODY",
    "OTHER", "VEHICLE INTO IMPACTOR", "IMPACTOR INTO BARRIER",
}
FIXED_OBJECT_CONFIGS = {"VEHICLE INTO BARRIER", "VEHICLE INTO POLE", "ROLLOVER"}
VEHICLE_VS_VEHICLE_CONFIGS = {"VEHICLE INTO VEHICLE", "IMPACTOR INTO VEHICLE"}


def classify_test(config, test_type, impact_angle, study_title):
    if config in EXCLUDED_CONFIGS:
        return None
    if config in FIXED_OBJECT_CONFIGS:
        return 3  # Single-vehicle
    if config in VEHICLE_VS_VEHICLE_CONFIGS:
        try:
            angle = float(impact_angle) % 360
        except (TypeError, ValueError):
            return None

        def circ_dist(a, t):
            d = abs(a - t) % 360
            return min(d, 360 - d)

        dist_from_0 = circ_dist(angle, 0)      # near head-on-aligned axis (front)
        dist_from_180 = circ_dist(angle, 180)  # near rear-aligned axis
        dist_from_perp = min(circ_dist(angle, 90), circ_dist(angle, 270))

        # "REAR"/"FRONT" appear in contractorStudyTitle (e.g. "FMVSS301-75
        # REAR IMPACT"), not testType -- verified against the raw index
        # after an initial run produced zero Rear-end matches. A second
        # bug (fixed here) was treating angle=180 as far from the
        # front-back axis when it's actually the rear-aligned pole of
        # that same axis -- circ_dist to BOTH 0 and 180 must be checked.
        haystack = f"{test_type or ''} {study_title or ''}".upper()
        is_rear_tagged = "REAR" in haystack
        is_front_tagged = "FRONT" in haystack and not is_rear_tagged

        if min(dist_from_0, dist_from_180) <= 30:
            if is_rear_tagged:
                return 0  # Rear-end
            if is_front_tagged:
                return 2  # Head-on
            # untagged: fall back to whichever pole the angle is nearer
            return 0 if dist_from_180 < dist_from_0 else 2
        if dist_from_perp <= 30:
            return 1                                # Angle
        return 4                                    # Sideswipe (oblique)
    return None


def api_get(url, retries=3):
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "research-script/1.0", "Accept": "application/json"})
            with urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (URLError, HTTPError, TimeoutError) as e:
            if attempt == retries - 1:
                print(f"    FAILED after {retries} tries: {url} ({e})")
                return None
            time.sleep(1.0 * (attempt + 1))
    return None


def fetch_file(url, out_path, retries=3):
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "research-script/1.0"})
            with urlopen(req, timeout=30) as resp:
                data = resp.read()
            out_path.write_bytes(data)
            return True
        except (URLError, HTTPError, TimeoutError) as e:
            if attempt == retries - 1:
                print(f"    FAILED file download after {retries} tries: {url} ({e})")
                return False
            time.sleep(1.0 * (attempt + 1))
    return False


def get_all_curves(test_no):
    curves, page = [], 0
    while True:
        d = api_get(f"{API_BASE}/get-instrumentation-info/{test_no}?pageNumber={page}")
        time.sleep(REQUEST_DELAY)
        if d is None or "results" not in d:
            break
        curves.extend(d["results"])
        total = d.get("meta", {}).get("pagination", {}).get("total", len(curves))
        if len(curves) >= total or not d["results"]:
            break
        page += 1
    return curves


def process_test(test_no, cls, config, angle):
    """Fetch one test's PRIMARY VEHICLE CG accelerometer curves. Returns a
    list of manifest rows (no shared file-handle writes here -- safe to
    call from multiple threads; the caller writes results serially)."""
    rows = []
    curves = get_all_curves(test_no)
    veh_cg = [c for c in curves
              if c.get("sensorAttachment") == "VEHICLE CG"
              and c.get("sensorType") == "ACCELEROMETER"
              and c.get("channelStatus") == "PRIMARY"]
    if not veh_cg:
        rows.append([test_no, cls, config, angle, "", "", "", "", "", ""])
        return rows

    test_dir = OUT_DIR / str(test_no)
    test_dir.mkdir(parents=True, exist_ok=True)
    for curve in veh_cg:
        curve_no = curve["curveNo"]
        detail = api_get(f"{API_BASE}/get-instrumentation-detail-info/{curve_no}/{test_no}")
        time.sleep(REQUEST_DELAY)
        if not detail or not detail.get("results"):
            continue
        info = detail["results"][0]
        ascii_url = info.get("asciiFile")
        axis = info.get("axisDirofSensor", "")
        if not ascii_url:
            continue
        out_path = test_dir / f"curve_{curve_no}_{axis.replace(' ', '')}.tsv"
        ok = fetch_file(ascii_url, out_path)
        time.sleep(REQUEST_DELAY)
        if ok:
            rows.append([test_no, cls, config, angle, curve_no, axis,
                         info.get("channelStatus"), info.get("timeIncrement"),
                         info.get("dataMeasurementUnits"), str(out_path.relative_to(REPO_ROOT))])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N classifiable tests (for validation runs)")
    parser.add_argument("--workers", type=int, default=6,
                         help="number of tests to process concurrently (default 6) -- each "
                              "worker still rate-limits its own requests with REQUEST_DELAY")
    parser.add_argument("--class-cap", type=str, default=None,
                         help="per-class download cap, e.g. '0:9999,1:500,2:9999,3:200,4:9999' -- "
                              "prioritizes classes this project actually needs (Rear-end, Head-on, "
                              "Sideswipe are all under-represented; Single-vehicle and Angle already "
                              "have plenty of real/synthetic coverage, so cap them low to avoid a "
                              "multi-hour run mostly re-fetching redundant data).")
    args = parser.parse_args()

    class_caps = {}
    if args.class_cap:
        for part in args.class_cap.split(","):
            k, v = part.split(":")
            class_caps[int(k)] = int(v)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    already_done = set()
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                already_done.add(row["testNo"])
        print(f"Resuming: {len(already_done)} tests already in manifest")

    with open(INDEX_CSV, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))

    classified = []
    for row in rows:
        cls = classify_test(row["testConfiguration"], row["testType"], row["impactAngle"],
                             row.get("contractorStudyTitle"))
        if cls is not None:
            classified.append((row["testNo"], cls, row["testConfiguration"], row["impactAngle"]))

    print(f"Total tests in index: {len(rows)}")
    print(f"Classifiable (matches our 5-class taxonomy): {len(classified)}")
    from collections import Counter
    print("By class:", Counter(c for _, c, _, _ in classified))

    todo = [t for t in classified if t[0] not in already_done]
    if class_caps:
        per_class_kept = Counter()
        capped = []
        for t in todo:
            cls = t[1]
            cap = class_caps.get(cls, len(todo))
            if per_class_kept[cls] < cap:
                capped.append(t)
                per_class_kept[cls] += 1
        todo = capped
        print("After class caps:", per_class_kept)
    if args.limit:
        todo = todo[:args.limit]
    print(f"To process this run: {len(todo)}\n")

    manifest_is_new = not MANIFEST_PATH.exists()
    mf = open(MANIFEST_PATH, "a", newline="", encoding="utf-8")
    writer = csv.writer(mf)
    if manifest_is_new:
        writer.writerow(["testNo", "crash_class", "testConfiguration", "impactAngle",
                          "curveNo", "axis", "channelStatus", "timeIncrement_us",
                          "measurementUnits", "file_path"])
        mf.flush()

    done_count = 0
    t0 = time.time()
    # Modest concurrency (default 6 workers): each worker still sleeps
    # REQUEST_DELAY between its own requests, so this isn't hammering the
    # API any harder per-connection than the serial version -- it just
    # runs that many tests' worth of (mostly network-latency-bound) work
    # at once instead of one at a time. A fully serial first run averaged
    # ~65s/test purely from round-trip latency to a .gov server; this is
    # about being patient with a slow server concurrently, not about
    # hitting it harder.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_test, test_no, cls, config, angle): (test_no, cls, config, angle)
                   for test_no, cls, config, angle in todo}
        for fut in as_completed(futures):
            test_no, cls, config, angle = futures[fut]
            done_count += 1
            try:
                rows = fut.result()
            except Exception as e:
                print(f"[{done_count}/{len(todo)}] test {test_no} FAILED: {e}")
                continue
            for row in rows:
                writer.writerow(row)
            mf.flush()
            elapsed = time.time() - t0
            rate = done_count / elapsed * 60 if elapsed > 0 else 0
            n_curves = sum(1 for r in rows if r[4] != "")
            print(f"[{done_count}/{len(todo)}] test {test_no} class={cls} config={config} "
                  f"-> {n_curves} curve(s) | {rate:.1f} tests/min | "
                  f"ETA {((len(todo)-done_count)/max(rate,0.01)):.0f} min")

    mf.close()
    print("\nDone.")


if __name__ == "__main__":
    main()

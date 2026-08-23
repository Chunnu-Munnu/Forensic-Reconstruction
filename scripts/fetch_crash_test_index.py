"""
Pulls the CURRENT live test index directly from the NHTSA Vehicle Crash
Test Database API (10,660 tests as of this run), replacing the earlier
Nov-2022 GitHub mirror (8,788 tests, stale) that
Dataset/data/raw/CrashTestDB/test_metadata_index.csv was built from.

Usage:
    .venv\\Scripts\\python.exe scripts\\fetch_crash_test_index.py
"""
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = REPO_ROOT / "Dataset" / "data" / "raw" / "CrashTestDB" / "test_metadata_index_live.csv"
API = "https://nrd.api.nhtsa.dot.gov/nhtsa/vehicle/api/v1/vehicle-database-test-results"
PAGE_SIZE = 20


def fetch_page(page):
    req = Request(f"{API}?pageNumber={page}", headers={"User-Agent": "research-script/1.0"})
    with urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    first = fetch_page(0)
    total = first["meta"]["pagination"]["total"]
    n_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    print(f"Total tests: {total}, pages: {n_pages}")

    all_rows = list(first["results"])
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(fetch_page, p) for p in range(1, n_pages)]
        for i, fut in enumerate(futures, 1):
            try:
                d = fut.result()
                all_rows.extend(d["results"])
            except Exception as e:
                print(f"page failed: {e}")
            if i % 50 == 0:
                print(f"  {i}/{n_pages} pages fetched, {len(all_rows)} rows so far")

    print(f"Total rows collected: {len(all_rows)}")

    fieldnames = ["testNo", "testReferenceNo", "testType", "testDate", "contractorStudyTitle",
                  "testPerformer", "impactAngle", "testConfiguration", "testConfigurationKey",
                  "offsetDistance", "closingSpeed"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"Saved to {OUT_CSV}")


if __name__ == "__main__":
    main()

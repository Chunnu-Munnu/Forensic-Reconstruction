"""
Splits a multi-member SAS XPORT V8 'library' file (NASS-CDS's .trx
distribution) into one valid single-member XPORT file per table, so
pandas/pyreadstat (neither of which respects member boundaries in a
library file, verified: both silently overrun into the next member's
bytes) can read each table correctly.
"""
import re
import sys
from pathlib import Path

MEMBER_MARKER = b"HEADER RECORD*******MEMBER  "
NAME_RE = re.compile(rb"SAS {5}([A-Z0-9_]+) *SASDATA")


def split(trx_path, out_dir):
    data = Path(trx_path).read_bytes()
    library_header = data[:240]

    offsets = []
    i = 0
    while True:
        i = data.find(MEMBER_MARKER, i)
        if i == -1:
            break
        offsets.append(i)
        i += 1
    offsets.append(len(data))  # sentinel for the last member's end

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for start, end in zip(offsets, offsets[1:]):
        chunk = data[start:end]
        m = NAME_RE.search(chunk[:400])
        name = m.group(1).decode("ascii").strip() if m else f"member_{start}"
        out_path = out_dir / f"{name}.xpt"
        out_path.write_bytes(library_header + chunk)
        written.append((name, out_path, len(chunk)))
    return written


if __name__ == "__main__":
    trx = sys.argv[1]
    out = sys.argv[2]
    for name, path, size in split(trx, out):
        print(f"{name:<12} {size:>10,} bytes -> {path}")

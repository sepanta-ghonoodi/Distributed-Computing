"""Download the NGSIM vehicle trajectory dataset.

    python scripts/download_ngsim.py

Public, no registration, no API key. The published CSV covers all four NGSIM
sites (US-101, I-80, Lankershim, Peachtree) and is roughly 1.5 GB / 11.8M rows;
`src.data.preprocess` filters it down to the two freeway sites.

Source: https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

NGSIM_URL = "https://data.transportation.gov/api/views/8ect-6jqj/rows.csv?accessType=DOWNLOAD"


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        bar = "#" * int(pct // 2.5)
        sys.stdout.write(
            f"\r  [{bar:<40}] {pct:5.1f}%  "
            f"{downloaded / 1e6:,.0f} / {total_size / 1e6:,.0f} MB"
        )
    else:
        sys.stdout.write(f"\r  {downloaded / 1e6:,.0f} MB")
    sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/ngsim_all.csv")
    ap.add_argument("--url", default=NGSIM_URL)
    ap.add_argument("--force", action="store_true", help="re-download if present")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        size = out.stat().st_size / 1e6
        print(f"{out} already present ({size:,.0f} MB) — skipping. Use --force to re-download.")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    # Download to a temporary name and rename only on success, so an interrupted
    # download can never be mistaken for a complete file on the next run.
    tmp = out.with_suffix(out.suffix + ".part")
    print(f"downloading NGSIM -> {out}")
    urllib.request.urlretrieve(args.url, tmp, reporthook=_progress)
    tmp.rename(out)
    print(f"\ndone | {out.stat().st_size / 1e6:,.0f} MB")


if __name__ == "__main__":
    main()

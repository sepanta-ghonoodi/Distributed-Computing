from __future__ import annotations
import argparse
import csv
import sys
import urllib.parse
import urllib.request
from pathlib import Path
RESOURCE = 'https://data.transportation.gov/resource/8ect-6jqj.csv'
FIELDS = ['vehicle_id', 'frame_id', 'global_time', 'local_x', 'local_y', 'v_length', 'v_width', 'v_vel', 'v_acc', 'lane_id', 'preceding', 'space_headway', 'location']

def _query(params: dict[str, str]) -> str:
    return RESOURCE + '?' + urllib.parse.urlencode(params)

def _fetch(url: str, timeout: int=300) -> list[list[str]]:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        text = r.read().decode('utf-8')
    return list(csv.reader(text.splitlines()))

def site_start_time(locations: list[str]) -> int:
    where = 'location in (' + ','.join((f"'{s}'" for s in locations)) + ')'
    rows = _fetch(_query({'$select': 'min(global_time)', '$where': where}))
    return int(float(rows[1][0]))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/raw/ngsim_us101.csv')
    ap.add_argument('--locations', nargs='+', default=['us-101'])
    ap.add_argument('--minutes', type=float, default=15.0, help='length of the time window to fetch (NGSIM records in 15-min periods)')
    ap.add_argument('--page', type=int, default=200000)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and (not args.force):
        print(f'{out} already present ({out.stat().st_size / 1000000.0:,.0f} MB) — skipping. Use --force to re-download.')
        return
    print(f'sites {args.locations}, first {args.minutes:g} minutes')
    t0 = site_start_time(args.locations)
    t1 = t0 + int(args.minutes * 60000)
    where = 'location in (' + ','.join((f"'{s}'" for s in args.locations)) + f') AND global_time >= {t0} AND global_time < {t1}'
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + '.part')
    seen: set[tuple[str, str]] = set()
    written = duplicates = offset = 0
    with open(tmp, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        while True:
            url = _query({'$select': ','.join(FIELDS), '$where': where, '$order': 'global_time,vehicle_id', '$limit': str(args.page), '$offset': str(offset)})
            rows = _fetch(url)
            body = rows[1:]
            if not body:
                break
            for row in body:
                key = (row[0], row[2])
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                writer.writerow(row)
                written += 1
            offset += args.page
            sys.stdout.write(f'\r  {written:,} rows  ({tmp.stat().st_size / 1000000.0:,.0f} MB)')
            sys.stdout.flush()
            if len(body) < args.page:
                break
    tmp.rename(out)
    print(f'\ndone | {out} | {written:,} rows, {out.stat().st_size / 1000000.0:,.0f} MB')
    if duplicates:
        print(f'  dropped {duplicates:,} duplicate (vehicle_id, global_time) rows')
    print(f'\nnext:\n  python -m src.data.preprocess --source ngsim --raw {out} --target-hz 2.0 --out data/processed/ngsim.parquet')
if __name__ == '__main__':
    main()

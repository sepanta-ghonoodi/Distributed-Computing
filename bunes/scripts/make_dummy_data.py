"""Generate a synthetic highway dataset so Phase 1 can run end-to-end today.

    python scripts/make_dummy_data.py

Equivalent to `python -m src.data.preprocess --source synthetic`, but with
knobs for the simulation size exposed directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import schema as S                       # noqa: E402
from src.data.preprocess import (                       # noqa: E402
    add_lane_offset,
    compute_kinematics,
    fill_leader_speed,
    resample,
)
from src.data.synthetic import generate_highway         # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicles", type=int, default=120)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--duration", type=float, default=600.0, help="seconds")
    ap.add_argument("--target-hz", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/processed/unified.parquet")
    args = ap.parse_args()

    print(f"simulating {args.vehicles} vehicles x {args.duration:g}s on {args.lanes} lanes ...")
    df = generate_highway(
        n_vehicles=args.vehicles,
        n_lanes=args.lanes,
        duration_s=args.duration,
        hz=10.0,
        seed=args.seed,
    )
    df = resample(df, source_hz=10.0, target_hz=args.target_hz)
    df = compute_kinematics(df, dt=1.0 / args.target_hz, smooth=True)
    df = fill_leader_speed(df)
    df = add_lane_offset(df)
    df = df.sort_values([S.VEHICLE_ID, S.FRAME]).reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(
        f"wrote {out} | {len(df):,} rows, {df[S.VEHICLE_ID].nunique()} vehicles, "
        f"{args.target_hz} Hz, mean speed {df[S.SPEED].mean():.1f} m/s"
    )


if __name__ == "__main__":
    main()

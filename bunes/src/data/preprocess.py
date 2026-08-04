"""Convert raw NGSIM / highD / synthetic data into the unified parquet.

Usage:
    python -m src.data.preprocess --source synthetic --out data/processed/unified.parquet
    python -m src.data.preprocess --source ngsim --raw data/raw/ngsim_us101.csv
    python -m src.data.preprocess --source highd  --raw data/raw/highd

Both real datasets are messy in their own way, so each parser is explicit about
what it assumes. If a column is missing the parser fails loudly rather than
silently producing NaNs that only show up as a bad ADE ten hours later.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema as S
from .synthetic import generate_highway
from .transforms import add_lane_offset, compute_kinematics


# ---------------------------------------------------------------------------
# NGSIM (US-101 / I-80), 10 Hz, units in feet
# ---------------------------------------------------------------------------
NGSIM_REQUIRED = [
    "Vehicle_ID", "Frame_ID", "Local_X", "Local_Y", "v_Length", "v_Width",
    "v_Vel", "v_Acc", "Lane_ID", "Preceding", "Space_Headway",
]


def load_ngsim(path: str | Path, hz: float = 10.0) -> pd.DataFrame:
    """Parse an NGSIM trajectory CSV into the unified schema."""
    df = pd.read_csv(path)
    missing = [c for c in NGSIM_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"NGSIM file {path} is missing columns: {missing}")

    # Vehicle_ID is only unique within a (Location, Direction) recording, so
    # build a globally unique id when those columns are present.
    key_cols = [c for c in ("Location", "Direction") if c in df.columns]
    if key_cols:
        prefix = df[key_cols].astype(str).agg("_".join, axis=1)
        vid = prefix + "_" + df["Vehicle_ID"].astype(str)
        vehicle_id = pd.factorize(vid)[0]
    else:
        vehicle_id = df["Vehicle_ID"].to_numpy()

    out = pd.DataFrame(
        {
            S.VEHICLE_ID: vehicle_id,
            S.FRAME: df["Frame_ID"].to_numpy(),
            # NGSIM: Local_Y is longitudinal, Local_X is lateral. Feet -> metres.
            S.X: df["Local_Y"].to_numpy() * S.FEET_TO_M,
            S.Y: df["Local_X"].to_numpy() * S.FEET_TO_M,
            S.SPEED: df["v_Vel"].to_numpy() * S.FEET_TO_M,
            S.ACCEL: df["v_Acc"].to_numpy() * S.FEET_TO_M,
            S.LANE_ID: df["Lane_ID"].to_numpy(),
            S.LENGTH: df["v_Length"].to_numpy() * S.FEET_TO_M,
            S.WIDTH: df["v_Width"].to_numpy() * S.FEET_TO_M,
            S.LEADER_ID: df["Preceding"].replace(0, -1).to_numpy(),
            S.GAP: df["Space_Headway"].replace(0, np.nan).to_numpy() * S.FEET_TO_M,
        }
    )
    # Re-base frames globally (not per vehicle): frame indices must stay on a
    # shared clock, otherwise the leader lookup in `fill_leader_speed` would
    # compare vehicles at different wall-clock times.
    out[S.FRAME] = out[S.FRAME] - out[S.FRAME].min()
    out[S.TIME] = out[S.FRAME] / hz
    out[S.LEADER_SPEED] = np.nan  # filled in below, after ids are unified
    for col in (S.VX, S.VY, S.HEADING):
        out[col] = np.nan
    return out[S.UNIFIED_COLUMNS]


# ---------------------------------------------------------------------------
# highD, 25 Hz, units already in metres
# ---------------------------------------------------------------------------
def load_highd(root: str | Path, recordings: list[int] | None = None) -> pd.DataFrame:
    """Parse highD `*_tracks.csv` + `*_recordingMeta.csv` into the unified schema.

    highD splits the two travel directions; vehicles driving in the negative-x
    direction are mirrored (x, y -> -x, -y) so that all traffic flows +x, which
    is what the rest of the pipeline assumes.
    """
    root = Path(root)
    track_files = sorted(root.glob("*_tracks.csv"))
    if not track_files:
        raise FileNotFoundError(f"No '*_tracks.csv' found under {root}")

    frames = []
    for tf in track_files:
        rec = int(tf.name.split("_")[0])
        if recordings is not None and rec not in recordings:
            continue
        meta_f = tf.with_name(tf.name.replace("_tracks.csv", "_tracksMeta.csv"))
        tracks = pd.read_csv(tf)
        meta = pd.read_csv(meta_f) if meta_f.exists() else None

        x = tracks["x"].to_numpy()
        y = tracks["y"].to_numpy()
        vx = tracks["xVelocity"].to_numpy()
        vy = tracks["yVelocity"].to_numpy()

        # Mirror the "upper" direction (drivingDirection == 1 travels -x).
        if meta is not None and "drivingDirection" in meta.columns:
            direction = tracks["id"].map(meta.set_index("id")["drivingDirection"])
            flip = (direction == 1).to_numpy()
            x = np.where(flip, -x, x)
            y = np.where(flip, -y, y)
            vx = np.where(flip, -vx, vx)
            vy = np.where(flip, -vy, vy)

        out = pd.DataFrame(
            {
                # Globally unique across recordings.
                S.VEHICLE_ID: rec * 100_000 + tracks["id"].to_numpy(),
                S.FRAME: tracks["frame"].to_numpy(),
                S.X: x,
                S.Y: y,
                S.VX: vx,
                S.VY: vy,
                S.SPEED: np.hypot(vx, vy),
                S.ACCEL: tracks["xAcceleration"].to_numpy(),
                S.LANE_ID: tracks["laneId"].to_numpy(),
                S.LENGTH: tracks["width"].to_numpy(),   # highD: 'width' is along x
                S.WIDTH: tracks["height"].to_numpy(),   # highD: 'height' is along y
                S.LEADER_ID: np.where(
                    tracks["precedingId"].to_numpy() > 0,
                    rec * 100_000 + tracks["precedingId"].to_numpy(),
                    -1,
                ),
                S.GAP: tracks["dhw"].replace(0, np.nan).to_numpy(),
                S.LEADER_SPEED: np.nan,
                S.HEADING: np.nan,
            }
        )
        out[S.TIME] = out[S.FRAME] / 25.0
        frames.append(out[S.UNIFIED_COLUMNS])

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Shared post-processing
# ---------------------------------------------------------------------------
def resample(df: pd.DataFrame, source_hz: float, target_hz: float) -> pd.DataFrame:
    """Decimate by an integer factor and renumber frames.

    Decimation (rather than interpolation) is intentional: highway trajectories
    are smooth at 10-25 Hz, so taking every k-th sample loses nothing, and it
    avoids inventing samples that the physics loss in Phase 3 would then be
    graded against.
    """
    factor = source_hz / target_hz
    if abs(factor - round(factor)) > 1e-6:
        raise ValueError(
            f"target_hz ({target_hz}) must divide source_hz ({source_hz}) evenly"
        )
    factor = int(round(factor))
    df = df.sort_values([S.VEHICLE_ID, S.FRAME])
    df = df[df[S.FRAME] % factor == 0].copy()
    df[S.FRAME] = df[S.FRAME] // factor
    df[S.TIME] = df[S.FRAME] / target_hz
    return df.reset_index(drop=True)


def fill_leader_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Look up each leader's speed at the same frame (needed by Phase 3's IDM)."""
    speeds = df.set_index([S.VEHICLE_ID, S.FRAME])[S.SPEED]
    key = pd.MultiIndex.from_arrays([df[S.LEADER_ID], df[S.FRAME]])
    df = df.copy()
    df[S.LEADER_SPEED] = speeds.reindex(key).to_numpy()
    return df


def build_unified(
    source: str,
    raw: str | Path | None,
    target_hz: float,
    seed: int = 0,
) -> pd.DataFrame:
    if source == "synthetic":
        df, source_hz = generate_highway(seed=seed), 10.0
    elif source == "ngsim":
        df, source_hz = load_ngsim(raw), 10.0
    elif source == "highd":
        df, source_hz = load_highd(raw), 25.0
    else:
        raise ValueError(f"Unknown source '{source}'")

    df = resample(df, source_hz, target_hz)
    df = compute_kinematics(df, dt=1.0 / target_hz, smooth=True)
    df = fill_leader_speed(df)
    df = add_lane_offset(df)
    return df.sort_values([S.VEHICLE_ID, S.FRAME]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Build the unified trajectory parquet.")
    p.add_argument("--source", choices=["synthetic", "ngsim", "highd"], default="synthetic")
    p.add_argument("--raw", default=None, help="Raw CSV file (NGSIM) or directory (highD)")
    p.add_argument("--target-hz", type=float, default=2.0)
    p.add_argument("--out", default="data/processed/unified.parquet")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = build_unified(args.source, args.raw, args.target_hz, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(
        f"Wrote {out}  |  {len(df):,} rows, "
        f"{df[S.VEHICLE_ID].nunique():,} vehicles, {args.target_hz} Hz"
    )


if __name__ == "__main__":
    main()

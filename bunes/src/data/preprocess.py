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
# The combined file published on data.transportation.gov covers all four NGSIM
# sites and does not use exactly the same column spelling as the older
# per-site releases, so every column is resolved case-insensitively against a
# list of known aliases rather than looked up by one hard-coded name.
NGSIM_ALIASES: dict[str, list[str]] = {
    "vehicle_id": ["vehicle_id"],
    "frame_id": ["frame_id"],
    "global_time": ["global_time"],
    "local_x": ["local_x"],
    "local_y": ["local_y"],
    "v_length": ["v_length", "v_len"],
    "v_width": ["v_width"],
    "v_vel": ["v_vel", "v_velocity"],
    "v_acc": ["v_acc", "v_acceleration"],
    "lane_id": ["lane_id"],
    "preceding": ["preceding", "preceeding"],
    "space_headway": ["space_headway", "space_hdwy"],
    "location": ["location"],
}

# Lankershim and Peachtree are urban arterials with signalised intersections.
# Training a highway model on them would be modelling the wrong thing entirely,
# so only the two freeway sites are kept by default.
NGSIM_FREEWAY_SITES = ["us-101", "i-80"]


def _resolve_ngsim_columns(columns) -> dict[str, str]:
    """Map canonical NGSIM field names onto whatever this file actually calls them."""
    lookup = {c.strip().lower(): c for c in columns}
    resolved: dict[str, str] = {}
    for canonical, aliases in NGSIM_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                resolved[canonical] = lookup[alias]
                break
    return resolved


def load_ngsim(
    path: str | Path,
    hz: float = 10.0,
    locations: list[str] | None = None,
    max_lane: int | None = 6,
) -> pd.DataFrame:
    """Parse an NGSIM trajectory CSV into the unified schema.

    Args:
        locations: site filter, matched case-insensitively against the
            `Location` column. Defaults to the two freeway sites.
        max_lane: drop lanes above this id. On US-101 lanes 1-5 are mainline
            and 6 is the auxiliary lane; 7-8 are the on/off ramps themselves,
            where "lane keeping" does not mean the same thing.
    """
    # Peek at the header first so the 1.5 GB body is read once, with only the
    # columns we need and without pandas guessing dtypes per chunk.
    header = pd.read_csv(path, nrows=0)
    cols = _resolve_ngsim_columns(header.columns)

    required = [
        "vehicle_id", "frame_id", "local_x", "local_y",
        "v_vel", "v_acc", "lane_id", "preceding", "space_headway",
    ]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(
            f"NGSIM file {path} is missing column(s) {missing}. "
            f"Found: {sorted(header.columns)}"
        )

    df = pd.read_csv(path, usecols=list(cols.values()), low_memory=False)
    print(f"  read {len(df):,} rows from {Path(path).name}")

    # --- site filter -------------------------------------------------------
    if "location" in cols:
        site = df[cols["location"]].astype(str).str.strip().str.lower()
        wanted = [s.lower() for s in (locations or NGSIM_FREEWAY_SITES)]
        keep = site.isin(wanted)
        present = sorted(site.unique())
        print(
            f"  sites present: {present} -> keeping {sorted(set(present) & set(wanted))}"
        )
        df, site = df[keep].copy(), site[keep]
        if df.empty:
            raise ValueError(f"No rows left after filtering to sites {wanted}")
    else:
        site = pd.Series("unknown", index=df.index)

    if max_lane is not None:
        before = len(df)
        df = df[df[cols["lane_id"]] <= max_lane].copy()
        site = site.loc[df.index]
        print(f"  lane filter (<= {max_lane}): {before:,} -> {len(df):,} rows")

    # --- clock -------------------------------------------------------------
    # Global_Time is epoch milliseconds and is the only field that puts every
    # site and every 15-minute recording period on one shared clock. Frame_ID
    # restarts per period, so using it directly would make vehicles from
    # different periods look simultaneous and corrupt the leader lookup.
    if "global_time" in cols:
        t_ms = df[cols["global_time"]].to_numpy(dtype=np.int64)
        frame = ((t_ms - t_ms.min()) / (1000.0 / hz)).round().astype(np.int64)
        period = (t_ms // (20 * 60 * 1000)).astype(np.int64)  # 20-min buckets
    else:
        frame = df[cols["frame_id"]].to_numpy(dtype=np.int64)
        frame = frame - frame.min()
        period = np.zeros(len(df), dtype=np.int64)

    # Vehicle_ID is unique only within one site and one recording period.
    # Built with pandas string concatenation rather than numpy `+`, which only
    # gained unicode support in numpy 2.0.
    period_s = pd.Series(period, index=df.index).astype(str)
    key = site.astype(str) + "_" + period_s + "_" + df[cols["vehicle_id"]].astype(str)
    vehicle_id = pd.factorize(key)[0]

    def col(name, default=np.nan):
        return df[cols[name]].to_numpy() if name in cols else np.full(len(df), default)

    out = pd.DataFrame(
        {
            S.VEHICLE_ID: vehicle_id,
            S.FRAME: frame,
            # NGSIM: Local_Y is longitudinal, Local_X is lateral. Feet -> metres.
            S.X: col("local_y") * S.FEET_TO_M,
            S.Y: col("local_x") * S.FEET_TO_M,
            S.SPEED: col("v_vel") * S.FEET_TO_M,
            S.ACCEL: col("v_acc") * S.FEET_TO_M,
            S.LANE_ID: col("lane_id"),
            S.LENGTH: col("v_length") * S.FEET_TO_M,
            S.WIDTH: col("v_width") * S.FEET_TO_M,
            S.LEADER_ID: np.where(col("preceding", 0) > 0, col("preceding", 0), -1),
            S.GAP: np.where(col("space_headway", 0) > 0, col("space_headway", 0), np.nan)
            * S.FEET_TO_M,
        }
    )
    out[S.TIME] = out[S.FRAME] / hz
    out[S.LEADER_SPEED] = np.nan  # filled in by fill_leader_speed
    for c in (S.VX, S.VY, S.HEADING):
        out[c] = np.nan

    # `preceding` refers to a raw Vehicle_ID within the same site+period, so it
    # has to be re-keyed the same way the ego ids were, or it points at nothing.
    preceding = df[cols["preceding"]].fillna(0).astype(np.int64)
    leader_key = site.astype(str) + "_" + period_s + "_" + preceding.astype(str)
    key_to_id = dict(zip(key.to_numpy(), vehicle_id))
    out[S.LEADER_ID] = np.where(
        preceding.to_numpy() > 0,
        [key_to_id.get(k, -1) for k in leader_key.to_numpy()],
        -1,
    )

    print(
        f"  -> {len(out):,} rows, {out[S.VEHICLE_ID].nunique():,} vehicles | "
        f"x {out[S.X].min():.0f}..{out[S.X].max():.0f} m, "
        f"y {out[S.Y].min():.1f}..{out[S.Y].max():.1f} m, "
        f"speed {out[S.SPEED].mean():.1f} m/s"
    )
    if out[S.X].max() > 3000:
        print("  [warn] longitudinal extent > 3 km — is this file already in metres?")

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
    locations: list[str] | None = None,
    max_lane: int | None = 6,
    max_vehicles: int | None = None,
) -> pd.DataFrame:
    if source == "synthetic":
        df, source_hz = generate_highway(seed=seed), 10.0
    elif source == "ngsim":
        df, source_hz = load_ngsim(raw, locations=locations, max_lane=max_lane), 10.0
    elif source == "highd":
        df, source_hz = load_highd(raw), 25.0
    else:
        raise ValueError(f"Unknown source '{source}'")

    # Decimate first: everything downstream loops over vehicles in Python, and
    # dropping 80% of the rows up front is the difference between one minute and
    # five on the full NGSIM file.
    df = resample(df, source_hz, target_hz)

    if max_vehicles is not None:
        rng = np.random.default_rng(seed)
        vehicles = df[S.VEHICLE_ID].unique()
        if len(vehicles) > max_vehicles:
            keep = rng.choice(vehicles, max_vehicles, replace=False)
            df = df[df[S.VEHICLE_ID].isin(keep)].copy()
            print(f"  subsampled to {max_vehicles:,} vehicles")

    print("  computing kinematics ...")
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
    p.add_argument(
        "--locations", nargs="+", default=None,
        help=f"NGSIM sites to keep (default: {NGSIM_FREEWAY_SITES})",
    )
    p.add_argument(
        "--max-lane", type=int, default=6,
        help="NGSIM: drop lanes above this id (ramps). Use -1 to keep all.",
    )
    p.add_argument("--max-vehicles", type=int, default=None)
    args = p.parse_args()

    df = build_unified(
        args.source, args.raw, args.target_hz, args.seed,
        locations=args.locations,
        max_lane=None if args.max_lane is not None and args.max_lane < 0 else args.max_lane,
        max_vehicles=args.max_vehicles,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(
        f"Wrote {out}  |  {len(df):,} rows, "
        f"{df[S.VEHICLE_ID].nunique():,} vehicles, {args.target_hz} Hz, "
        f"lanes {sorted(df[S.LANE_ID].unique())[:10]}"
    )


if __name__ == "__main__":
    main()

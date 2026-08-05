from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from . import schema as S
from .synthetic import generate_highway
from .transforms import add_lane_offset, compute_kinematics
NGSIM_ALIASES: dict[str, list[str]] = {'vehicle_id': ['vehicle_id'], 'frame_id': ['frame_id'], 'global_time': ['global_time'], 'local_x': ['local_x'], 'local_y': ['local_y'], 'v_length': ['v_length', 'v_len'], 'v_width': ['v_width'], 'v_vel': ['v_vel', 'v_velocity'], 'v_acc': ['v_acc', 'v_acceleration'], 'lane_id': ['lane_id'], 'preceding': ['preceding', 'preceeding'], 'space_headway': ['space_headway', 'space_hdwy'], 'location': ['location']}
NGSIM_FREEWAY_SITES = ['us-101', 'i-80']

def _resolve_ngsim_columns(columns) -> dict[str, str]:
    lookup = {c.strip().lower(): c for c in columns}
    resolved: dict[str, str] = {}
    for (canonical, aliases) in NGSIM_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                resolved[canonical] = lookup[alias]
                break
    return resolved

def load_ngsim(path: str | Path, hz: float=10.0, locations: list[str] | None=None, max_lane: int | None=6) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    cols = _resolve_ngsim_columns(header.columns)
    required = ['vehicle_id', 'frame_id', 'local_x', 'local_y', 'v_vel', 'v_acc', 'lane_id', 'preceding', 'space_headway']
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f'NGSIM file {path} is missing column(s) {missing}. Found: {sorted(header.columns)}')
    df = pd.read_csv(path, usecols=list(cols.values()), low_memory=False)
    print(f'  read {len(df):,} rows from {Path(path).name}')
    if 'location' in cols:
        site = df[cols['location']].astype(str).str.strip().str.lower()
        wanted = [s.lower() for s in locations or NGSIM_FREEWAY_SITES]
        keep = site.isin(wanted)
        present = sorted(site.unique())
        print(f'  sites present: {present} -> keeping {sorted(set(present) & set(wanted))}')
        (df, site) = (df[keep].copy(), site[keep])
        if df.empty:
            raise ValueError(f'No rows left after filtering to sites {wanted}')
    else:
        site = pd.Series('unknown', index=df.index)
    if max_lane is not None:
        before = len(df)
        df = df[df[cols['lane_id']] <= max_lane].copy()
        site = site.loc[df.index]
        print(f'  lane filter (<= {max_lane}): {before:,} -> {len(df):,} rows')
    if 'global_time' in cols:
        t_ms = df[cols['global_time']].to_numpy(dtype=np.int64)
        frame = ((t_ms - t_ms.min()) / (1000.0 / hz)).round().astype(np.int64)
        period = (t_ms // (20 * 60 * 1000)).astype(np.int64)
    else:
        frame = df[cols['frame_id']].to_numpy(dtype=np.int64)
        frame = frame - frame.min()
        period = np.zeros(len(df), dtype=np.int64)
    period_s = pd.Series(period, index=df.index).astype(str)
    key = site.astype(str) + '_' + period_s + '_' + df[cols['vehicle_id']].astype(str)
    vehicle_id = pd.factorize(key)[0]

    def col(name, default=np.nan):
        return df[cols[name]].to_numpy() if name in cols else np.full(len(df), default)
    out = pd.DataFrame({S.VEHICLE_ID: vehicle_id, S.FRAME: frame, S.X: col('local_y') * S.FEET_TO_M, S.Y: col('local_x') * S.FEET_TO_M, S.SPEED: col('v_vel') * S.FEET_TO_M, S.ACCEL: col('v_acc') * S.FEET_TO_M, S.LANE_ID: col('lane_id'), S.LENGTH: col('v_length') * S.FEET_TO_M, S.WIDTH: col('v_width') * S.FEET_TO_M, S.LEADER_ID: np.where(col('preceding', 0) > 0, col('preceding', 0), -1), S.GAP: np.where(col('space_headway', 0) > 0, col('space_headway', 0), np.nan) * S.FEET_TO_M})
    out[S.TIME] = out[S.FRAME] / hz
    out[S.LEADER_SPEED] = np.nan
    for c in (S.VX, S.VY, S.HEADING):
        out[c] = np.nan
    preceding = df[cols['preceding']].fillna(0).astype(np.int64)
    leader_key = site.astype(str) + '_' + period_s + '_' + preceding.astype(str)
    key_to_id = dict(zip(key.to_numpy(), vehicle_id))
    out[S.LEADER_ID] = np.where(preceding.to_numpy() > 0, [key_to_id.get(k, -1) for k in leader_key.to_numpy()], -1)
    print(f'  -> {len(out):,} rows, {out[S.VEHICLE_ID].nunique():,} vehicles | x {out[S.X].min():.0f}..{out[S.X].max():.0f} m, y {out[S.Y].min():.1f}..{out[S.Y].max():.1f} m, speed {out[S.SPEED].mean():.1f} m/s')
    if out[S.X].max() > 3000:
        print('  [warn] longitudinal extent > 3 km — is this file already in metres?')
    return out[S.UNIFIED_COLUMNS]

def load_highd(root: str | Path, recordings: list[int] | None=None) -> pd.DataFrame:
    root = Path(root)
    track_files = sorted(root.glob('*_tracks.csv'))
    if not track_files:
        raise FileNotFoundError(f"No '*_tracks.csv' found under {root}")
    frames = []
    for tf in track_files:
        rec = int(tf.name.split('_')[0])
        if recordings is not None and rec not in recordings:
            continue
        meta_f = tf.with_name(tf.name.replace('_tracks.csv', '_tracksMeta.csv'))
        tracks = pd.read_csv(tf)
        meta = pd.read_csv(meta_f) if meta_f.exists() else None
        x = tracks['x'].to_numpy()
        y = tracks['y'].to_numpy()
        vx = tracks['xVelocity'].to_numpy()
        vy = tracks['yVelocity'].to_numpy()
        if meta is not None and 'drivingDirection' in meta.columns:
            direction = tracks['id'].map(meta.set_index('id')['drivingDirection'])
            flip = (direction == 1).to_numpy()
            x = np.where(flip, -x, x)
            y = np.where(flip, -y, y)
            vx = np.where(flip, -vx, vx)
            vy = np.where(flip, -vy, vy)
        out = pd.DataFrame({S.VEHICLE_ID: rec * 100000 + tracks['id'].to_numpy(), S.FRAME: tracks['frame'].to_numpy(), S.X: x, S.Y: y, S.VX: vx, S.VY: vy, S.SPEED: np.hypot(vx, vy), S.ACCEL: tracks['xAcceleration'].to_numpy(), S.LANE_ID: tracks['laneId'].to_numpy(), S.LENGTH: tracks['width'].to_numpy(), S.WIDTH: tracks['height'].to_numpy(), S.LEADER_ID: np.where(tracks['precedingId'].to_numpy() > 0, rec * 100000 + tracks['precedingId'].to_numpy(), -1), S.GAP: tracks['dhw'].replace(0, np.nan).to_numpy(), S.LEADER_SPEED: np.nan, S.HEADING: np.nan})
        out[S.TIME] = out[S.FRAME] / 25.0
        frames.append(out[S.UNIFIED_COLUMNS])
    return pd.concat(frames, ignore_index=True)

def resample(df: pd.DataFrame, source_hz: float, target_hz: float) -> pd.DataFrame:
    factor = source_hz / target_hz
    if abs(factor - round(factor)) > 1e-06:
        raise ValueError(f'target_hz ({target_hz}) must divide source_hz ({source_hz}) evenly')
    factor = int(round(factor))
    df = df.sort_values([S.VEHICLE_ID, S.FRAME])
    df = df[df[S.FRAME] % factor == 0].copy()
    df[S.FRAME] = df[S.FRAME] // factor
    df[S.TIME] = df[S.FRAME] / target_hz
    return df.reset_index(drop=True)

def fill_leader_speed(df: pd.DataFrame) -> pd.DataFrame:
    speeds = df.set_index([S.VEHICLE_ID, S.FRAME])[S.SPEED]
    key = pd.MultiIndex.from_arrays([df[S.LEADER_ID], df[S.FRAME]])
    df = df.copy()
    df[S.LEADER_SPEED] = speeds.reindex(key).to_numpy()
    return df

def build_unified(source: str, raw: str | Path | None, target_hz: float, seed: int=0, locations: list[str] | None=None, max_lane: int | None=6, max_vehicles: int | None=None) -> pd.DataFrame:
    if source == 'synthetic':
        (df, source_hz) = (generate_highway(seed=seed), 10.0)
    elif source == 'ngsim':
        (df, source_hz) = (load_ngsim(raw, locations=locations, max_lane=max_lane), 10.0)
    elif source == 'highd':
        (df, source_hz) = (load_highd(raw), 25.0)
    else:
        raise ValueError(f"Unknown source '{source}'")
    df = resample(df, source_hz, target_hz)
    if max_vehicles is not None:
        rng = np.random.default_rng(seed)
        vehicles = df[S.VEHICLE_ID].unique()
        if len(vehicles) > max_vehicles:
            keep = rng.choice(vehicles, max_vehicles, replace=False)
            df = df[df[S.VEHICLE_ID].isin(keep)].copy()
            print(f'  subsampled to {max_vehicles:,} vehicles')
    print('  computing kinematics ...')
    df = compute_kinematics(df, dt=1.0 / target_hz, smooth=True)
    df = fill_leader_speed(df)
    df = add_lane_offset(df)
    return df.sort_values([S.VEHICLE_ID, S.FRAME]).reset_index(drop=True)

def main() -> None:
    p = argparse.ArgumentParser(description='Build the unified trajectory parquet.')
    p.add_argument('--source', choices=['synthetic', 'ngsim', 'highd'], default='synthetic')
    p.add_argument('--raw', default=None, help='Raw CSV file (NGSIM) or directory (highD)')
    p.add_argument('--target-hz', type=float, default=2.0)
    p.add_argument('--out', default='data/processed/unified.parquet')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--locations', nargs='+', default=None, help=f'NGSIM sites to keep (default: {NGSIM_FREEWAY_SITES})')
    p.add_argument('--max-lane', type=int, default=6, help='NGSIM: drop lanes above this id (ramps). Use -1 to keep all.')
    p.add_argument('--max-vehicles', type=int, default=None)
    args = p.parse_args()
    df = build_unified(args.source, args.raw, args.target_hz, args.seed, locations=args.locations, max_lane=None if args.max_lane is not None and args.max_lane < 0 else args.max_lane, max_vehicles=args.max_vehicles)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f'Wrote {out}  |  {len(df):,} rows, {df[S.VEHICLE_ID].nunique():,} vehicles, {args.target_hz} Hz, lanes {sorted(df[S.LANE_ID].unique())[:10]}')
if __name__ == '__main__':
    main()

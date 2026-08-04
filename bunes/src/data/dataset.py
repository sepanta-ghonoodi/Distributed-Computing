"""Sliding-window construction and the PyTorch Dataset.

Every window is expressed in an *agent frame* anchored at the last observed
point (origin) and the heading at that point (rotation). Two consequences:

  * The model never sees absolute highway coordinates, so it generalises across
    road sections instead of memorising them.
  * The targets are small, zero-centred displacements, which is far better
    conditioned than raw metres and removes most of the trivial drift.

`origin` and `theta` are kept in every batch so Phase 2 can convert a partial
rollout back to world coordinates, snap it to a lane centreline, and convert the
correction back into the agent frame.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset

from ..config import DataConfig
from . import schema as S
from .transforms import Scaler, to_agent_frame


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------
@dataclass
class WindowBundle:
    """All windows of one split, as plain numpy arrays."""

    features: np.ndarray    # (N, obs_len, F)  agent-frame, *unscaled*
    fut_delta: np.ndarray   # (N, pred_len, 2) agent-frame per-step displacement
    fut_pos: np.ndarray     # (N, pred_len, 2) agent-frame absolute position
    origin: np.ndarray      # (N, 2)  world position of the agent-frame origin
    theta: np.ndarray       # (N,)    world heading of the agent-frame +x axis
    vehicle_id: np.ndarray  # (N,)
    start_frame: np.ndarray  # (N,)
    # Leader state at the last observed step, for the Phase 3 IDM regulariser.
    # NaN where the vehicle has no preceding vehicle.
    leader_gap: np.ndarray   # (N,)  bumper-to-bumper gap [m]
    leader_speed: np.ndarray  # (N,) preceding vehicle speed [m/s]
    desired_speed: np.ndarray  # (N,) free-flow speed target [m/s]

    def __len__(self) -> int:
        return len(self.features)

    def subset(self, mask: np.ndarray) -> "WindowBundle":
        return WindowBundle(
            **{f: getattr(self, f)[mask] for f in self.__dataclass_fields__}
        )


def _contiguous_runs(frames: np.ndarray) -> list[tuple[int, int]]:
    """Split an ascending frame index into runs with no gaps: [(start, stop), ...]."""
    breaks = np.where(np.diff(frames) != 1)[0]
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks + 1, [len(frames)]])
    return list(zip(starts.tolist(), stops.tolist()))


def build_windows(df: pd.DataFrame, cfg: DataConfig) -> WindowBundle:
    """Slice the unified dataframe into fixed-length observation/prediction windows."""
    obs_len, pred_len = cfg.obs_len, cfg.pred_len
    total_len = obs_len + pred_len

    if "lane_offset" not in df.columns:
        df = df.assign(lane_offset=0.0)

    vehicles = df[S.VEHICLE_ID].unique()
    if cfg.max_vehicles is not None and len(vehicles) > cfg.max_vehicles:
        rng = np.random.default_rng(cfg.split_seed)
        vehicles = rng.choice(vehicles, cfg.max_vehicles, replace=False)
        df = df[df[S.VEHICLE_ID].isin(vehicles)]

    chunks: list[np.ndarray] = []
    meta_vid: list[np.ndarray] = []
    meta_frame: list[np.ndarray] = []

    cols = [S.X, S.Y, S.SPEED, S.ACCEL, S.HEADING, "lane_offset", S.GAP, S.LEADER_SPEED]
    for vid, g in df.sort_values([S.VEHICLE_ID, S.FRAME]).groupby(S.VEHICLE_ID, sort=False):
        frames = g[S.FRAME].to_numpy()
        raw = g[cols].to_numpy(dtype=np.float64)
        for lo, hi in _contiguous_runs(frames):
            seg = raw[lo:hi]
            if len(seg) < total_len:
                continue
            # (n_windows, total_len, n_cols) view — no copy until we index it.
            win = sliding_window_view(seg, total_len, axis=0)          # (n0, C, L)
            win = np.moveaxis(win, -1, 1)[:: cfg.window_stride]        # (n,  L, C)
            chunks.append(win.copy())

            n0 = len(seg) - total_len + 1                              # windows before striding
            starts = frames[lo:hi][:n0][:: cfg.window_stride]
            meta_vid.append(np.full(len(starts), vid))
            meta_frame.append(starts)

    if not chunks:
        raise RuntimeError(
            f"No windows of length {total_len} could be built. "
            "Check target_hz / obs_seconds / pred_seconds against the data length."
        )

    win = np.concatenate(chunks, 0)                    # (N, L, C)
    vehicle_id = np.concatenate(meta_vid, 0)
    start_frame = np.concatenate(meta_frame, 0)

    # --- drop near-stationary windows (no useful long-horizon signal) ------
    # NaN-safe: gap/leader_speed are NaN whenever there is no preceding vehicle,
    # and nanmean would otherwise be needed everywhere downstream.
    keep = win[:, :, 2].mean(1) >= cfg.min_mean_speed
    win, vehicle_id, start_frame = win[keep], vehicle_id[keep], start_frame[keep]

    xy = win[:, :, 0:2]                                # (N, L, 2) world
    speed = win[:, :, 2]
    accel = win[:, :, 3]
    heading = win[:, :, 4]
    lane_offset = win[:, :, 5]

    # --- agent frame -------------------------------------------------------
    origin = xy[:, obs_len - 1, :]                     # last observed position
    theta = heading[:, obs_len - 1]                    # heading at that instant
    xy_a = to_agent_frame(xy, origin, theta)           # (N, L, 2)

    # Per-step displacement. The first step has no predecessor inside the
    # window, so we repeat the second one rather than injecting a spurious zero.
    delta = np.diff(xy_a, axis=1)
    delta = np.concatenate([delta[:, :1, :], delta], axis=1)   # (N, L, 2)

    rel_heading = np.arctan2(
        np.sin(heading - theta[:, None]), np.cos(heading - theta[:, None])
    )

    features = np.stack(
        [
            xy_a[:, :obs_len, 0],       # rel_x
            xy_a[:, :obs_len, 1],       # rel_y
            delta[:, :obs_len, 0],      # delta_x
            delta[:, :obs_len, 1],      # delta_y
            speed[:, :obs_len],
            accel[:, :obs_len],
            np.sin(rel_heading[:, :obs_len]),
            np.cos(rel_heading[:, :obs_len]),
            lane_offset[:, :obs_len],
        ],
        axis=-1,
    ).astype(np.float32)
    assert features.shape[-1] == S.NUM_FEATURES

    fut_pos = xy_a[:, obs_len:, :].astype(np.float32)          # (N, pred_len, 2)
    # delta[t] = pos[t] - pos[t-1], with pos[-1] == origin == (0, 0) by construction.
    fut_delta = np.diff(
        np.concatenate([np.zeros_like(fut_pos[:, :1, :]), fut_pos], axis=1), axis=1
    ).astype(np.float32)

    # --- leader state at the last observed step (Phase 3) ------------------
    # The desired free-flow speed is not observable, so it is approximated by
    # the fastest the ego actually travelled during the observation window —
    # a vehicle held up behind a slow leader still reveals its preferred speed
    # in the moments before it closed the gap.
    desired_speed = np.maximum(speed[:, :obs_len].max(axis=1), 5.0)

    return WindowBundle(
        features=features,
        fut_delta=fut_delta,
        fut_pos=fut_pos,
        origin=origin.astype(np.float32),
        theta=theta.astype(np.float32),
        vehicle_id=vehicle_id,
        start_frame=start_frame,
        leader_gap=win[:, obs_len - 1, 6].astype(np.float32),
        leader_speed=win[:, obs_len - 1, 7].astype(np.float32),
        desired_speed=desired_speed.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_by_vehicle(bundle: WindowBundle, cfg: DataConfig) -> dict[str, WindowBundle]:
    """Partition windows by vehicle id so no vehicle appears in two splits."""
    vehicles = np.unique(bundle.vehicle_id)
    rng = np.random.default_rng(cfg.split_seed)
    rng.shuffle(vehicles)

    n = len(vehicles)
    n_test = int(round(n * cfg.test_fraction))
    n_val = int(round(n * cfg.val_fraction))
    groups = {
        "test": set(vehicles[:n_test].tolist()),
        "val": set(vehicles[n_test : n_test + n_val].tolist()),
        "train": set(vehicles[n_test + n_val :].tolist()),
    }
    return {
        name: bundle.subset(np.isin(bundle.vehicle_id, list(ids)))
        for name, ids in groups.items()
    }


# ---------------------------------------------------------------------------
# Torch dataset
# ---------------------------------------------------------------------------
class HighwayWindowDataset(Dataset):
    """Yields one training window.

    Batch contents:
        src        (obs_len, F)   scaled encoder input
        tgt_delta  (pred_len, 2)  ground-truth per-step displacement [m]
        tgt_pos    (pred_len, 2)  ground-truth absolute position, agent frame [m]
        cv_delta   (2,)           last observed displacement [m] (CV prior)
        origin     (2,)  theta ()  world pose of the agent frame (for Phase 2/6)
    """

    def __init__(self, bundle: WindowBundle, scaler: Scaler):
        self.b = bundle
        self.scaler = scaler
        self.src = scaler.transform(bundle.features)
        # Unscaled last observed displacement (feature channels delta_x/delta_y
        # at the final observed timestep) — the constant-velocity prior.
        dx = S.FEATURE_NAMES.index("delta_x")
        self.cv_delta = bundle.features[:, -1, dx : dx + 2].copy()

    def __len__(self) -> int:
        return len(self.b)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        # The model derives its own teacher-forcing tokens from `tgt_pos` and
        # `cv_delta`, so no shifted input needs to be materialised here.
        return {
            "src": torch.from_numpy(self.src[i]),
            "tgt_delta": torch.from_numpy(self.b.fut_delta[i]),
            "tgt_pos": torch.from_numpy(self.b.fut_pos[i]),
            "cv_delta": torch.from_numpy(self.cv_delta[i]),
            "origin": torch.from_numpy(self.b.origin[i]),
            "theta": torch.tensor(self.b.theta[i]),
            # NaN (no leader) survives to the loss, which masks on it.
            "leader_gap": torch.tensor(self.b.leader_gap[i]),
            "leader_speed": torch.tensor(self.b.leader_speed[i]),
            "desired_speed": torch.tensor(self.b.desired_speed[i]),
        }


# ---------------------------------------------------------------------------
# Entry point with on-disk caching
# ---------------------------------------------------------------------------
# Bumped whenever WindowBundle gains or changes a field, so a stale cache from
# an earlier schema fails to be reused rather than loading with missing keys.
_CACHE_VERSION = 2


def _cache_key(cfg: DataConfig) -> str:
    relevant = {"_v": _CACHE_VERSION} | {
        k: getattr(cfg, k)
        for k in (
            "unified_path", "target_hz", "obs_seconds", "pred_seconds",
            "window_stride", "min_mean_speed", "val_fraction", "test_fraction",
            "split_seed", "max_vehicles",
        )
    }
    return hashlib.md5(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:12]


def load_splits(
    cfg: DataConfig, use_cache: bool = True
) -> tuple[dict[str, HighwayWindowDataset], Scaler]:
    """Build (or load) train/val/test datasets plus the fitted feature scaler."""
    cache = Path(cfg.cache_dir) / f"windows_{_cache_key(cfg)}.npz"

    if use_cache and cache.exists():
        blob = np.load(cache, allow_pickle=True)
        splits = {
            name: WindowBundle(
                **{f: blob[f"{name}__{f}"] for f in WindowBundle.__dataclass_fields__}
            )
            for name in ("train", "val", "test")
        }
        scaler = Scaler.from_dict(json.loads(blob["scaler"].item()))
    else:
        df = pd.read_parquet(cfg.unified_path)
        bundle = build_windows(df, cfg)
        splits = split_by_vehicle(bundle, cfg)
        # Fit the scaler on training windows only — fitting on everything leaks
        # test-set statistics into training.
        scaler = Scaler.fit(splits["train"].features)

        cache.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            f"{name}__{f}": getattr(b, f)
            for name, b in splits.items()
            for f in WindowBundle.__dataclass_fields__
        }
        payload["scaler"] = json.dumps(scaler.to_dict())
        np.savez_compressed(cache, **payload)

    datasets = {n: HighwayWindowDataset(b, scaler) for n, b in splits.items()}
    return datasets, scaler

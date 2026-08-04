"""Kinematics derivation, agent-frame geometry, and feature scaling.

The agent-frame transform is the piece that matters most for later phases:
the model is trained entirely in a translated+rotated frame anchored at the last
observed point, but Phase 2 (Link Projection) must snap predictions to a *world*
lane centreline. So the forward and inverse transforms are exposed as a small,
explicitly invertible pair rather than being inlined in the dataset code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from . import schema as S


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------
def compute_kinematics(df: pd.DataFrame, dt: float, smooth: bool = True) -> pd.DataFrame:
    """Fill vx/vy/speed/accel/heading by differentiating x/y per vehicle.

    NGSIM's published velocity/acceleration channels are notoriously noisy
    (they were differentiated from a noisy position signal without filtering),
    so we recompute them from lightly smoothed positions instead.

    Args:
        df: unified-schema dataframe, sorted or unsorted.
        dt: seconds between consecutive frames.
        smooth: apply a Savitzky-Golay filter to x/y before differentiating.
    """
    # np.gradient needs at least two samples. Real data has vehicles that enter
    # the camera's field of view for a fraction of a second, or that survive
    # resampling as a single frame; the synthetic simulator has none, which is
    # why this only surfaced on NGSIM. Such tracks carry no usable kinematics
    # and are far shorter than one window, so drop them here rather than
    # special-casing them through every downstream stage.
    counts = df.groupby(S.VEHICLE_ID)[S.FRAME].transform("size")
    too_short = counts < 3
    if bool(too_short.any()):
        n_veh = df.loc[too_short, S.VEHICLE_ID].nunique()
        print(
            f"  dropping {n_veh:,} vehicle(s) with < 3 frames "
            f"({int(too_short.sum()):,} rows) — too short to differentiate"
        )
        df = df[~too_short]
    if df.empty:
        raise ValueError("No vehicle has enough frames to compute kinematics")

    out = []
    for vid, g in df.sort_values([S.VEHICLE_ID, S.FRAME]).groupby(S.VEHICLE_ID, sort=False):
        g = g.copy()
        x = g[S.X].to_numpy(dtype=np.float64)
        y = g[S.Y].to_numpy(dtype=np.float64)

        if smooth and len(x) >= 9:
            # window must be odd and <= signal length
            win = min(9, len(x) if len(x) % 2 == 1 else len(x) - 1)
            x = savgol_filter(x, win, polyorder=2)
            y = savgol_filter(y, win, polyorder=2)
            g[S.X], g[S.Y] = x, y

        # Central differences (np.gradient) rather than diff: no half-step lag.
        vx = np.gradient(x, dt)
        vy = np.gradient(y, dt)
        speed = np.hypot(vx, vy)
        accel = np.gradient(speed, dt)

        # Heading from the velocity vector; falls back to "straight ahead" when
        # the vehicle is essentially stopped and the direction is meaningless.
        heading = np.arctan2(vy, vx)
        heading[speed < 0.1] = 0.0

        g[S.VX], g[S.VY] = vx, vy
        g[S.SPEED], g[S.ACCEL], g[S.HEADING] = speed, accel, heading
        out.append(g)

    return pd.concat(out, ignore_index=True)


def add_lane_offset(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `lane_offset` column: lateral distance from the own-lane centre.

    Lane centres are estimated as the median lateral position of every sample
    recorded in that lane. This is a stand-in for the real lane geometry, which
    Phase 2 replaces with actual centreline LineStrings.
    """
    df = df.copy()
    centres = df.groupby(S.LANE_ID)[S.Y].median()
    df["lane_offset"] = df[S.Y] - df[S.LANE_ID].map(centres).astype(float)
    df["lane_offset"] = df["lane_offset"].fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Agent frame
# ---------------------------------------------------------------------------
def rotation_matrix(theta: np.ndarray) -> np.ndarray:
    """Batched 2x2 rotation matrices. `theta` shape (B,) -> (B, 2, 2)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def to_agent_frame(points: np.ndarray, origin: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """World -> agent frame.

    Args:
        points: (B, T, 2) world coordinates.
        origin: (B, 2) translation (the last observed position).
        theta:  (B,)   rotation (the heading at the last observed step).
    Returns:
        (B, T, 2) coordinates with the origin at `origin` and +x along `theta`.
    """
    rel = points - origin[:, None, :]
    r_inv = rotation_matrix(-theta)                    # (B, 2, 2)
    return np.einsum("bij,btj->bti", r_inv, rel)


def from_agent_frame(points: np.ndarray, origin: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Agent frame -> world. Exact inverse of `to_agent_frame`."""
    r = rotation_matrix(theta)
    return np.einsum("bij,btj->bti", r, points) + origin[:, None, :]


def rotate_vectors(vectors: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Rotate free vectors (velocities, displacements) — no translation."""
    r_inv = rotation_matrix(-theta)
    return np.einsum("bij,btj->bti", r_inv, vectors)


# ---------------------------------------------------------------------------
# Feature scaling
# ---------------------------------------------------------------------------
@dataclass
class Scaler:
    """Per-channel standardisation, fitted on the training split only."""

    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def fit(x: np.ndarray, eps: float = 1e-6) -> "Scaler":
        """`x` of shape (..., C); statistics taken over every leading axis."""
        flat = x.reshape(-1, x.shape[-1])
        return Scaler(
            mean=flat.mean(0).astype(np.float32),
            std=(flat.std(0) + eps).astype(np.float32),
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_dict(d: dict) -> "Scaler":
        return Scaler(
            mean=np.asarray(d["mean"], dtype=np.float32),
            std=np.asarray(d["std"], dtype=np.float32),
        )

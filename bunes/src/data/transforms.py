from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from . import schema as S

def compute_kinematics(df: pd.DataFrame, dt: float, smooth: bool=True) -> pd.DataFrame:
    counts = df.groupby(S.VEHICLE_ID)[S.FRAME].transform('size')
    too_short = counts < 3
    if bool(too_short.any()):
        n_veh = df.loc[too_short, S.VEHICLE_ID].nunique()
        print(f'  dropping {n_veh:,} vehicle(s) with < 3 frames ({int(too_short.sum()):,} rows) — too short to differentiate')
        df = df[~too_short]
    if df.empty:
        raise ValueError('No vehicle has enough frames to compute kinematics')
    out = []
    for (vid, g) in df.sort_values([S.VEHICLE_ID, S.FRAME]).groupby(S.VEHICLE_ID, sort=False):
        g = g.copy()
        x = g[S.X].to_numpy(dtype=np.float64)
        y = g[S.Y].to_numpy(dtype=np.float64)
        if smooth and len(x) >= 9:
            win = min(9, len(x) if len(x) % 2 == 1 else len(x) - 1)
            x = savgol_filter(x, win, polyorder=2)
            y = savgol_filter(y, win, polyorder=2)
            (g[S.X], g[S.Y]) = (x, y)
        vx = np.gradient(x, dt)
        vy = np.gradient(y, dt)
        speed = np.hypot(vx, vy)
        accel = np.gradient(speed, dt)
        heading = np.arctan2(vy, vx)
        heading[speed < 0.1] = 0.0
        (g[S.VX], g[S.VY]) = (vx, vy)
        (g[S.SPEED], g[S.ACCEL], g[S.HEADING]) = (speed, accel, heading)
        out.append(g)
    return pd.concat(out, ignore_index=True)

def add_lane_offset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    centres = df.groupby(S.LANE_ID)[S.Y].median()
    df['lane_offset'] = df[S.Y] - df[S.LANE_ID].map(centres).astype(float)
    df['lane_offset'] = df['lane_offset'].fillna(0.0)
    return df

def rotation_matrix(theta: np.ndarray) -> np.ndarray:
    (c, s) = (np.cos(theta), np.sin(theta))
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)

def to_agent_frame(points: np.ndarray, origin: np.ndarray, theta: np.ndarray) -> np.ndarray:
    rel = points - origin[:, None, :]
    r_inv = rotation_matrix(-theta)
    return np.einsum('bij,btj->bti', r_inv, rel)

def from_agent_frame(points: np.ndarray, origin: np.ndarray, theta: np.ndarray) -> np.ndarray:
    r = rotation_matrix(theta)
    return np.einsum('bij,btj->bti', r, points) + origin[:, None, :]

def rotate_vectors(vectors: np.ndarray, theta: np.ndarray) -> np.ndarray:
    r_inv = rotation_matrix(-theta)
    return np.einsum('bij,btj->bti', r_inv, vectors)

@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def fit(x: np.ndarray, eps: float=1e-06) -> 'Scaler':
        flat = x.reshape(-1, x.shape[-1])
        return Scaler(mean=flat.mean(0).astype(np.float32), std=(flat.std(0) + eps).astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict:
        return {'mean': self.mean.tolist(), 'std': self.std.tolist()}

    @staticmethod
    def from_dict(d: dict) -> 'Scaler':
        return Scaler(mean=np.asarray(d['mean'], dtype=np.float32), std=np.asarray(d['std'], dtype=np.float32))

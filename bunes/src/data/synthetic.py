from __future__ import annotations
import numpy as np
import pandas as pd
from . import schema as S
LANE_WIDTH = 3.7
IDM_A_MAX = 1.4
IDM_B_COMF = 2.0
IDM_S0 = 2.0
IDM_T_HEADWAY = 1.5
IDM_DELTA = 4.0

def idm_acceleration(v: np.ndarray, v0: np.ndarray, gap: np.ndarray, dv: np.ndarray) -> np.ndarray:
    gap = np.maximum(gap, 0.1)
    s_star = IDM_S0 + np.maximum(0.0, v * IDM_T_HEADWAY + v * dv / (2.0 * np.sqrt(IDM_A_MAX * IDM_B_COMF)))
    free = 1.0 - (v / np.maximum(v0, 0.1)) ** IDM_DELTA
    interaction = (s_star / gap) ** 2
    return IDM_A_MAX * (free - interaction)

def _smoothstep(u: np.ndarray | float) -> np.ndarray | float:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)

def generate_highway(n_vehicles: int=120, n_lanes: int=4, duration_s: float=600.0, hz: float=10.0, road_length: float=2500.0, lane_change_rate: float=0.03, lane_change_duration: float=3.5, seed: int=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt = 1.0 / hz
    n_steps = int(round(duration_s * hz))
    length = rng.uniform(4.0, 5.2, n_vehicles)
    length[rng.random(n_vehicles) < 0.12] = rng.uniform(12.0, 16.5)
    width = np.where(length > 8.0, 2.5, rng.uniform(1.7, 1.95, n_vehicles))
    v_desired = rng.normal(31.0, 2.8, n_vehicles).clip(22.0, 38.0)
    v_desired[length > 8.0] *= 0.85
    lane = rng.integers(0, n_lanes, n_vehicles)
    x = np.zeros(n_vehicles)
    for ln in range(n_lanes):
        idx = np.where(lane == ln)[0]
        if len(idx):
            x[idx] = np.sort(rng.uniform(0, road_length, len(idx)))
    v = v_desired * rng.uniform(0.85, 1.0, n_vehicles)
    x_total = x.copy()
    y = lane * LANE_WIDTH + rng.normal(0, 0.15, n_vehicles)
    lc_active = np.zeros(n_vehicles, dtype=bool)
    lc_from = lane.copy()
    lc_to = lane.copy()
    lc_time = np.zeros(n_vehicles)
    records = []
    for step in range(n_steps):
        gap = np.full(n_vehicles, np.inf)
        leader_id = np.full(n_vehicles, -1, dtype=int)
        leader_v = np.full(n_vehicles, np.nan)
        occupancy_lane = np.where(lc_active, lc_to, lane)
        for ln in range(n_lanes):
            idx = np.where(occupancy_lane == ln)[0]
            if len(idx) < 2:
                continue
            order = idx[np.argsort(x[idx])]
            ahead = np.roll(order, -1)
            raw_gap = (x[ahead] - x[order]) % road_length
            gap[order] = raw_gap - length[ahead]
            leader_id[order] = ahead
            leader_v[order] = v[ahead]
        dv = np.where(np.isfinite(gap), v - np.nan_to_num(leader_v, nan=0.0), 0.0)
        a = idm_acceleration(v, v_desired, gap, dv)
        a = np.clip(a, -6.0, IDM_A_MAX)
        held_up = np.isfinite(gap) & (gap < 45.0) & (v < 0.92 * v_desired)
        wants = held_up & ~lc_active & (rng.random(n_vehicles) < lane_change_rate * dt)
        for i in np.where(wants)[0]:
            candidates = [c for c in (lane[i] - 1, lane[i] + 1) if 0 <= c < n_lanes]
            rng.shuffle(candidates)
            for c in candidates:
                others = np.where((occupancy_lane == c) & (np.arange(n_vehicles) != i))[0]
                if len(others) == 0:
                    target_ok = True
                else:
                    d = (x[others] - x[i] + road_length / 2) % road_length - road_length / 2
                    target_ok = bool(np.all(np.abs(d) > 25.0))
                if target_ok:
                    lc_active[i] = True
                    lc_from[i] = lane[i]
                    lc_to[i] = c
                    lc_time[i] = 0.0
                    break
        v = np.maximum(0.0, v + a * dt)
        x = (x + v * dt) % road_length
        x_total = x_total + v * dt
        lc_time = np.where(lc_active, lc_time + dt, lc_time)
        progress = np.where(lc_active, _smoothstep(lc_time / lane_change_duration), 0.0)
        y_target = np.where(lc_active, (lc_from + (lc_to - lc_from) * progress) * LANE_WIDTH, lane * LANE_WIDTH)
        wander = rng.normal(0, 0.02, n_vehicles)
        y = y + 0.35 * (y_target - y) + wander
        finished = lc_active & (lc_time >= lane_change_duration)
        lane = np.where(finished, lc_to, lane)
        lc_active = lc_active & ~finished
        records.append(pd.DataFrame({S.VEHICLE_ID: np.arange(n_vehicles), S.FRAME: step, S.TIME: step * dt, S.X: x_total, S.Y: y, S.SPEED: v, S.ACCEL: a, S.LANE_ID: lane, S.LENGTH: length, S.WIDTH: width, S.LEADER_ID: leader_id, S.GAP: np.where(np.isfinite(gap), gap, np.nan), S.LEADER_SPEED: leader_v}))
    df = pd.concat(records, ignore_index=True)
    df[S.VX] = np.nan
    df[S.VY] = np.nan
    df[S.HEADING] = np.nan
    return df[S.UNIFIED_COLUMNS]

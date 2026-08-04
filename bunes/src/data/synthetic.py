"""Synthetic highway traffic generator (dummy data for Phase 1).

Produces trajectories in the unified schema so the whole pipeline can be
exercised before NGSIM/highD is downloaded. Longitudinal motion follows the
Intelligent Driver Model — the same model Phase 3 turns into a loss term — so
the dummy data is physically consistent rather than random noise, and the
Phase 3 physics regulariser will have something meaningful to grade against.

Lane changes use a smoothstep lateral profile, giving the curved, laterally
non-trivial trajectories that make Phase 2's Link Projection worth having.

Simulation runs on a ring of length `road_length` to keep density constant;
the reported x is *unwrapped* (monotonically increasing) so no trajectory ever
contains an artificial jump at the wrap point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema as S

LANE_WIDTH = 3.7  # m, standard US freeway lane


# --- IDM parameters ---------------------------------------------------------
IDM_A_MAX = 1.4      # max acceleration [m/s^2]
IDM_B_COMF = 2.0     # comfortable deceleration [m/s^2]
IDM_S0 = 2.0         # minimum bumper-to-bumper gap [m]
IDM_T_HEADWAY = 1.5  # desired time headway [s]
IDM_DELTA = 4.0      # acceleration exponent


def idm_acceleration(
    v: np.ndarray, v0: np.ndarray, gap: np.ndarray, dv: np.ndarray
) -> np.ndarray:
    """Intelligent Driver Model acceleration.

    Args:
        v:    current speed [m/s]
        v0:   desired free-flow speed [m/s]
        gap:  bumper-to-bumper distance to the leader [m] (inf if no leader)
        dv:   approach rate, v_ego - v_leader [m/s]
    """
    gap = np.maximum(gap, 0.1)
    s_star = IDM_S0 + np.maximum(
        0.0, v * IDM_T_HEADWAY + (v * dv) / (2.0 * np.sqrt(IDM_A_MAX * IDM_B_COMF))
    )
    free = 1.0 - (v / np.maximum(v0, 0.1)) ** IDM_DELTA
    interaction = (s_star / gap) ** 2
    return IDM_A_MAX * (free - interaction)


def _smoothstep(u: np.ndarray | float) -> np.ndarray | float:
    """C1-continuous 0->1 ramp; used for the lateral lane-change profile."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def generate_highway(
    n_vehicles: int = 120,
    n_lanes: int = 4,
    duration_s: float = 600.0,
    hz: float = 10.0,
    road_length: float = 2500.0,
    lane_change_rate: float = 0.03,    # hazard rate [1/s] while "held up"
    lane_change_duration: float = 3.5,  # seconds
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate a multi-lane highway and return a unified-schema dataframe."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / hz
    n_steps = int(round(duration_s * hz))

    # --- vehicle properties ---------------------------------------------
    length = rng.uniform(4.0, 5.2, n_vehicles)
    length[rng.random(n_vehicles) < 0.12] = rng.uniform(12.0, 16.5)  # trucks
    width = np.where(length > 8.0, 2.5, rng.uniform(1.7, 1.95, n_vehicles))
    v_desired = rng.normal(31.0, 2.8, n_vehicles).clip(22.0, 38.0)   # ~110 km/h
    # Trucks are slower and keep to the right-hand lanes.
    v_desired[length > 8.0] *= 0.85

    # --- initial state ---------------------------------------------------
    lane = rng.integers(0, n_lanes, n_vehicles)
    # Spread vehicles evenly per lane to avoid overlapping spawns.
    x = np.zeros(n_vehicles)
    for ln in range(n_lanes):
        idx = np.where(lane == ln)[0]
        if len(idx):
            x[idx] = np.sort(rng.uniform(0, road_length, len(idx)))
    v = v_desired * rng.uniform(0.85, 1.0, n_vehicles)
    x_total = x.copy()                     # unwrapped longitudinal position
    y = lane * LANE_WIDTH + rng.normal(0, 0.15, n_vehicles)

    # Lane-change state: (active, from_lane, to_lane, progress_seconds)
    lc_active = np.zeros(n_vehicles, dtype=bool)
    lc_from = lane.copy()
    lc_to = lane.copy()
    lc_time = np.zeros(n_vehicles)

    records = []
    for step in range(n_steps):
        # ---- find the leader of each vehicle within its (target) lane ----
        gap = np.full(n_vehicles, np.inf)
        leader_id = np.full(n_vehicles, -1, dtype=int)
        leader_v = np.full(n_vehicles, np.nan)
        occupancy_lane = np.where(lc_active, lc_to, lane)  # commit to target lane

        for ln in range(n_lanes):
            idx = np.where(occupancy_lane == ln)[0]
            if len(idx) < 2:
                continue
            order = idx[np.argsort(x[idx])]
            ahead = np.roll(order, -1)  # ring: the last vehicle follows the first
            raw_gap = (x[ahead] - x[order]) % road_length
            gap[order] = raw_gap - length[ahead]
            leader_id[order] = ahead
            leader_v[order] = v[ahead]

        dv = np.where(np.isfinite(gap), v - np.nan_to_num(leader_v, nan=0.0), 0.0)
        a = idm_acceleration(v, v_desired, gap, dv)
        a = np.clip(a, -6.0, IDM_A_MAX)

        # ---- discretionary lane changes ---------------------------------
        # A vehicle considers changing lane when it is being held up by a slow
        # leader; the target lane must have a comfortable gap.
        held_up = np.isfinite(gap) & (gap < 45.0) & (v < 0.92 * v_desired)
        wants = held_up & (~lc_active) & (rng.random(n_vehicles) < lane_change_rate * dt)
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

        # ---- integrate --------------------------------------------------
        v = np.maximum(0.0, v + a * dt)
        x = (x + v * dt) % road_length
        x_total = x_total + v * dt

        lc_time = np.where(lc_active, lc_time + dt, lc_time)
        progress = np.where(lc_active, _smoothstep(lc_time / lane_change_duration), 0.0)
        y_target = np.where(
            lc_active,
            (lc_from + (lc_to - lc_from) * progress) * LANE_WIDTH,
            lane * LANE_WIDTH,
        )
        # First-order lag towards the target keeps the lateral signal smooth
        # even outside a lane change (lane-keeping wander).
        wander = rng.normal(0, 0.02, n_vehicles)
        y = y + 0.35 * (y_target - y) + wander

        finished = lc_active & (lc_time >= lane_change_duration)
        lane = np.where(finished, lc_to, lane)
        lc_active = lc_active & ~finished

        records.append(
            pd.DataFrame(
                {
                    S.VEHICLE_ID: np.arange(n_vehicles),
                    S.FRAME: step,
                    S.TIME: step * dt,
                    S.X: x_total,
                    S.Y: y,
                    S.SPEED: v,
                    S.ACCEL: a,
                    S.LANE_ID: lane,
                    S.LENGTH: length,
                    S.WIDTH: width,
                    S.LEADER_ID: leader_id,
                    S.GAP: np.where(np.isfinite(gap), gap, np.nan),
                    S.LEADER_SPEED: leader_v,
                }
            )
        )

    df = pd.concat(records, ignore_index=True)
    # vx/vy/heading are derived by the shared kinematics routine so synthetic
    # and real data go through exactly the same code path.
    df[S.VX] = np.nan
    df[S.VY] = np.nan
    df[S.HEADING] = np.nan
    return df[S.UNIFIED_COLUMNS]

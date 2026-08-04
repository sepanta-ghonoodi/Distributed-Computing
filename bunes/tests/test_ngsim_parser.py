"""Tests for the NGSIM parser, run against a hand-built miniature file.

The real download is 1.5 GB, so the parser is exercised here on a few dozen
rows that reproduce the properties that actually break it: the combined file's
column spelling, four sites mixed together, Vehicle_ID colliding across
recording periods, ramp lanes, and feet rather than metres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import schema as S
from src.data.preprocess import load_ngsim

FT = 1.0 / 0.3048
PERIOD_MS = 20 * 60 * 1000  # the bucket width the parser uses to separate periods


def _row(vid, t_ms, x_ft, y_ft, lane, site, preceding=0, headway_ft=0.0):
    return {
        "Vehicle_ID": vid,
        "Frame_ID": (t_ms // 100) % 10_000,   # restarts per period, like the real file
        "Global_Time": t_ms,
        "Local_X": x_ft,                       # lateral
        "Local_Y": y_ft,                       # longitudinal
        "v_length": 15.0,
        "v_Width": 6.0,
        "v_Vel": 60.0,
        "v_Acc": 0.5,
        "Lane_ID": lane,
        "Preceding": preceding,
        "Space_Headway": headway_ft,
        "Location": site,
    }


@pytest.fixture
def mini_csv(tmp_path):
    rows = []
    base = 1_113_433_000_000  # arbitrary epoch ms

    # Two freeway sites, plus an arterial that must be dropped.
    for site in ("us-101", "i-80", "lankershim"):
        for k in range(10):
            rows.append(_row(1, base + k * 100, 20.0 * FT, 100.0 * FT + k * 30, 2, site,
                             preceding=2, headway_ft=80.0 * FT))
            rows.append(_row(2, base + k * 100, 20.0 * FT, 180.0 * FT + k * 30, 2, site))

    # Same Vehicle_ID=1 in a *different* recording period at the same site.
    for k in range(10):
        rows.append(_row(1, base + PERIOD_MS * 3 + k * 100, 24.0 * FT, 50.0 * FT + k * 30,
                         3, "us-101"))

    # A ramp lane that the default max_lane filter should remove.
    for k in range(10):
        rows.append(_row(9, base + k * 100, 40.0 * FT, 90.0 * FT + k * 30, 8, "us-101"))

    path = tmp_path / "ngsim_mini.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_drops_arterial_sites(mini_csv):
    df = load_ngsim(mini_csv)
    # 3 sites x 2 vehicles x 10 frames = 60 rows, minus lankershim's 20,
    # plus the second-period vehicle's 10. Ramp lane 8 is filtered out.
    assert len(df) == 50


def test_drops_ramp_lanes(mini_csv):
    assert load_ngsim(mini_csv)[S.LANE_ID].max() <= 6
    assert 8 in load_ngsim(mini_csv, max_lane=None)[S.LANE_ID].unique()


def test_vehicle_ids_do_not_collide_across_sites_or_periods(mini_csv):
    df = load_ngsim(mini_csv)
    # Vehicle_ID 1 appears at us-101, at i-80, and again at us-101 in a later
    # period. All three must become distinct ids, or their trajectories get
    # concatenated into one impossible vehicle.
    assert df[S.VEHICLE_ID].nunique() == 5


def test_frames_share_one_clock(mini_csv):
    """Frame_ID restarts per period; the parser must key off Global_Time.

    If it did not, the later period's frames would overlap the first period's
    and the leader lookup would pair up vehicles that were never on the road at
    the same time.
    """
    df = load_ngsim(mini_csv)
    per_period_max = df.groupby(S.VEHICLE_ID)[S.FRAME].max()
    # The second-period vehicle must sit far beyond the first period's frames.
    assert per_period_max.max() > 3 * PERIOD_MS / 100 - 1


def test_units_and_axis_convention(mini_csv):
    df = load_ngsim(mini_csv)
    # Local_Y (longitudinal, feet) -> x in metres.
    assert 10.0 < df[S.X].min() < 60.0
    # Local_X (lateral, feet) -> y in metres.
    assert np.isclose(df[S.Y].min(), 20.0, atol=1.0)
    # 60 ft/s ~ 18.3 m/s
    assert np.isclose(df[S.SPEED].iloc[0], 60.0 * 0.3048, atol=0.01)


def test_leader_ids_are_remapped(mini_csv):
    """`Preceding` holds a raw Vehicle_ID; it must be re-keyed like the ego ids."""
    df = load_ngsim(mini_csv)
    followers = df[df[S.LEADER_ID] >= 0]
    assert len(followers) > 0
    # Every referenced leader must actually exist in the parsed frame.
    assert set(followers[S.LEADER_ID]).issubset(set(df[S.VEHICLE_ID]))
    # ... and must never be the vehicle itself.
    assert not (followers[S.LEADER_ID] == followers[S.VEHICLE_ID]).any()


def test_missing_columns_fail_loudly(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Vehicle_ID": [1], "Frame_ID": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing column"):
        load_ngsim(path)

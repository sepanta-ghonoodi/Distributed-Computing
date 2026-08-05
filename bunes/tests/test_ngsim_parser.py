from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.data import schema as S
from src.data.preprocess import load_ngsim
FT = 1.0 / 0.3048
PERIOD_MS = 20 * 60 * 1000

def _row(vid, t_ms, x_ft, y_ft, lane, site, preceding=0, headway_ft=0.0):
    return {'Vehicle_ID': vid, 'Frame_ID': t_ms // 100 % 10000, 'Global_Time': t_ms, 'Local_X': x_ft, 'Local_Y': y_ft, 'v_length': 15.0, 'v_Width': 6.0, 'v_Vel': 60.0, 'v_Acc': 0.5, 'Lane_ID': lane, 'Preceding': preceding, 'Space_Headway': headway_ft, 'Location': site}

@pytest.fixture
def mini_csv(tmp_path):
    rows = []
    base = 1113433000000
    for site in ('us-101', 'i-80', 'lankershim'):
        for k in range(10):
            rows.append(_row(1, base + k * 100, 20.0 * FT, 100.0 * FT + k * 30, 2, site, preceding=2, headway_ft=80.0 * FT))
            rows.append(_row(2, base + k * 100, 20.0 * FT, 180.0 * FT + k * 30, 2, site))
    for k in range(10):
        rows.append(_row(1, base + PERIOD_MS * 3 + k * 100, 24.0 * FT, 50.0 * FT + k * 30, 3, 'us-101'))
    for k in range(10):
        rows.append(_row(9, base + k * 100, 40.0 * FT, 90.0 * FT + k * 30, 8, 'us-101'))
    path = tmp_path / 'ngsim_mini.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

def test_drops_arterial_sites(mini_csv):
    df = load_ngsim(mini_csv)
    assert len(df) == 50

def test_drops_ramp_lanes(mini_csv):
    assert load_ngsim(mini_csv)[S.LANE_ID].max() <= 6
    assert 8 in load_ngsim(mini_csv, max_lane=None)[S.LANE_ID].unique()

def test_vehicle_ids_do_not_collide_across_sites_or_periods(mini_csv):
    df = load_ngsim(mini_csv)
    assert df[S.VEHICLE_ID].nunique() == 5

def test_frames_share_one_clock(mini_csv):
    df = load_ngsim(mini_csv)
    per_period_max = df.groupby(S.VEHICLE_ID)[S.FRAME].max()
    assert per_period_max.max() > 3 * PERIOD_MS / 100 - 1

def test_units_and_axis_convention(mini_csv):
    df = load_ngsim(mini_csv)
    assert 10.0 < df[S.X].min() < 60.0
    assert np.isclose(df[S.Y].min(), 20.0, atol=1.0)
    assert np.isclose(df[S.SPEED].iloc[0], 60.0 * 0.3048, atol=0.01)

def test_leader_ids_are_remapped(mini_csv):
    df = load_ngsim(mini_csv)
    followers = df[df[S.LEADER_ID] >= 0]
    assert len(followers) > 0
    assert set(followers[S.LEADER_ID]).issubset(set(df[S.VEHICLE_ID]))
    assert not (followers[S.LEADER_ID] == followers[S.VEHICLE_ID]).any()

def test_missing_columns_fail_loudly(tmp_path):
    path = tmp_path / 'bad.csv'
    pd.DataFrame({'Vehicle_ID': [1], 'Frame_ID': [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='missing column'):
        load_ngsim(path)

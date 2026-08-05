from __future__ import annotations
import torch
from src.vec.migration import migration_metrics, reactive_metrics
from src.vec.rsu import RSUChain, first_handover, to_world

def test_rsu_index_is_a_floor_division():
    chain = RSUChain(spacing=300.0, x0=0.0)
    x = torch.tensor([0.0, 299.9, 300.0, 601.0])
    assert chain.index(x).tolist() == [0, 0, 1, 2]

def test_first_handover_finds_the_crossing_step():
    chain = RSUChain(spacing=300.0, x0=0.0)
    dt = 0.5
    start = 250.0
    x = start + 10.0 * torch.arange(1, 21, dtype=torch.float32) * dt
    x = x.unsqueeze(0)
    (t, occurred) = first_handover(x, chain.index(torch.tensor([start])), chain, dt)
    assert bool(occurred[0])
    assert abs(float(t[0]) - 5.0) < 1e-05

def test_no_handover_is_reported_not_faked():
    chain = RSUChain(spacing=300.0, x0=0.0)
    x = (10.0 + 0.5 * torch.arange(1, 21, dtype=torch.float32)).unsqueeze(0)
    (t, occurred) = first_handover(x, chain.index(torch.tensor([10.0])), chain, 0.5)
    assert not bool(occurred[0])
    assert float(t[0]) == 0.0

def test_perfect_prediction_removes_the_interruption():
    t_true = torch.tensor([4.0, 9.0, 12.0])
    m = migration_metrics(t_true.clone(), torch.ones(3, dtype=torch.bool), t_true, 3.0)
    assert m['mean_interruption_s'] == 0.0
    assert m['zero_interruption_rate'] == 1.0
    assert m['interruption_reduction_pct'] == 100.0

def test_reactive_always_pays_the_full_migration_time():
    t_true = torch.tensor([4.0, 9.0])
    assert reactive_metrics(t_true, 3.0)['mean_interruption_s'] == 3.0

def test_late_prediction_is_penalised_but_capped():
    t_true = torch.tensor([10.0, 10.0])
    t_pred = torch.tensor([11.0, 110.0])
    m = migration_metrics(t_pred, torch.ones(2, dtype=torch.bool), t_true, 3.0)
    assert abs(m['mean_interruption_s'] - 2.0) < 1e-06

def test_early_prediction_costs_residency_not_interruption():
    t_true = torch.tensor([10.0])
    m = migration_metrics(torch.tensor([6.0]), torch.ones(1, dtype=torch.bool), t_true, 3.0)
    assert m['mean_interruption_s'] == 0.0
    assert abs(m['mean_premature_s'] - 4.0) < 1e-06

def test_missing_a_handover_degrades_to_reactive():
    t_true = torch.tensor([10.0, 10.0])
    m = migration_metrics(torch.zeros(2), torch.zeros(2, dtype=torch.bool), t_true, 3.0)
    assert m['mean_interruption_s'] == 3.0
    assert m['handover_detect_rate'] == 0.0

def test_margin_trades_interruption_for_residency():
    t_true = torch.tensor([10.0, 10.0])
    t_pred = torch.tensor([12.0, 12.0])
    no_margin = migration_metrics(t_pred, torch.ones(2, dtype=torch.bool), t_true, 3.0)
    with_margin = migration_metrics(t_pred, torch.ones(2, dtype=torch.bool), t_true, 3.0, margin=2.0)
    assert abs(no_margin['mean_interruption_s'] - 2.0) < 1e-06
    assert with_margin['mean_interruption_s'] == 0.0
    assert with_margin['mean_premature_s'] == 0.0
    generous = migration_metrics(t_pred, torch.ones(2, dtype=torch.bool), t_true, 3.0, margin=5.0)
    assert generous['mean_interruption_s'] == 0.0
    assert abs(generous['mean_premature_s'] - 3.0) < 1e-06

def test_margin_cannot_rescue_an_undetected_handover():
    t_true = torch.tensor([10.0])
    m = migration_metrics(torch.zeros(1), torch.zeros(1, dtype=torch.bool), t_true, 3.0, margin=5.0)
    assert m['mean_interruption_s'] == 3.0

def test_to_world_round_trips_the_agent_frame():
    from src.map.highway import to_agent_frame_t
    (n, t) = (5, 7)
    pos = torch.randn(n, t, 2) * 50
    origin = torch.randn(n, 2) * 400
    theta = torch.rand(n) * 0.4 - 0.2
    world = to_world(pos, origin, theta)
    back = to_agent_frame_t(world.reshape(n * t, 2), origin.repeat_interleave(t, 0), theta.repeat_interleave(t, 0)).reshape(n, t, 2)
    assert torch.allclose(back, pos, atol=0.0001)
